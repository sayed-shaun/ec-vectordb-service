"""Qdrant persistence.

One collection, one point per chunk, one named vector per configured model,
so a chunk can never be missing under just one of them.
"""

from __future__ import annotations

import logging
import uuid

from qdrant_client import QdrantClient, models

from src.core.config import MODEL_SPECS, settings
from src.fusion import Candidate

log = logging.getLogger(__name__)


def stable_point_id(doc_id: str) -> str:
    """Hash a doc_id into a deterministic UUID, as Qdrant ids must be uint64
    or UUID. Re-ingesting a doc_id overwrites in place."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"ec-vectordb/{doc_id}"))


class VectorStore:
    def __init__(self, client: QdrantClient | None = None, collection: str | None = None) -> None:
        self.collection = collection or settings.COLLECTION
        self.client = client or QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=60,
        )

    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            return
        log.info("creating collection %s", self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                spec.name: models.VectorParams(
                    size=spec.dim,
                    distance=models.Distance.COSINE,
                    hnsw_config=models.HnswConfigDiff(m=16, ef_construct=128),
                )
                for spec in MODEL_SPECS
            },
        )
        for field_name in ("source", "lang", "doc_ref"):
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    def upsert(
        self,
        doc_ids: list[str],
        texts: list[str],
        vectors_by_model: dict[str, list[list[float]]],
        metadatas: list[dict] | None = None,
    ) -> int:
        metadatas = metadatas or [{} for _ in doc_ids]
        points = []
        for i, doc_id in enumerate(doc_ids):
            payload = {"doc_id": doc_id, "text": texts[i], **metadatas[i]}
            points.append(
                models.PointStruct(
                    id=stable_point_id(doc_id),
                    vector={name: vecs[i] for name, vecs in vectors_by_model.items()},
                    payload=payload,
                )
            )
        self.client.upsert(collection_name=self.collection, points=points, wait=True)
        return len(points)

    def search_all(
        self,
        vectors_by_model: dict[str, list[float]],
        limit: int,
        query_filter: models.Filter | None = None,
    ) -> dict[str, list[Candidate]]:
        """Run all model searches in one round trip, so each extra model
        costs almost no latency over one."""
        names = list(vectors_by_model)
        requests = [
            models.QueryRequest(
                query=vectors_by_model[name],
                using=name,
                limit=limit,
                filter=query_filter,
                with_payload=True,
            )
            for name in names
        ]
        responses = self.client.query_batch_points(
            collection_name=self.collection, requests=requests
        )
        out: dict[str, list[Candidate]] = {}
        for name, response in zip(names, responses):
            out[name] = [
                Candidate(
                    doc_id=h.payload.get("doc_id", str(h.id)),
                    score=float(h.score),
                    rank=i + 1,
                    payload=h.payload or {},
                )
                for i, h in enumerate(response.points)
            ]
        return out

    def delete(self, doc_ids: list[str]) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.PointIdsList(points=[stable_point_id(d) for d in doc_ids]),
            wait=True,
        )

    def prune(self, keep_doc_ids: set[str]) -> int:
        """Delete every point whose doc_id is not in ``keep_doc_ids``.

        A reindex upserts in place, so queries keep working throughout; this
        runs last to drop points a shrunk or reshuffled corpus no longer has.
        """
        keep = {stable_point_id(d) for d in keep_doc_ids}
        stale: list = []
        offset = None
        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection,
                limit=settings.INDEX_BATCH_SIZE,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            stale.extend(r.id for r in records if str(r.id) not in keep)
            if offset is None:
                break
        for i in range(0, len(stale), settings.INDEX_BATCH_SIZE):
            self.client.delete(
                collection_name=self.collection,
                points_selector=models.PointIdsList(
                    points=stale[i : i + settings.INDEX_BATCH_SIZE]
                ),
                wait=True,
            )
        return len(stale)

    def existing_doc_ids(self, doc_ids: list[str]) -> set[str]:
        """Which of these doc_ids are already indexed, so a long ingest can
        resume instead of restarting."""
        found: set[str] = set()
        step = 256
        for i in range(0, len(doc_ids), step):
            chunk = doc_ids[i : i + step]
            records = self.client.retrieve(
                collection_name=self.collection,
                ids=[stable_point_id(d) for d in chunk],
                with_payload=["doc_id"],
                with_vectors=False,
            )
            found.update(
                r.payload.get("doc_id") for r in records if r.payload and r.payload.get("doc_id")
            )
        return found

    def count(self) -> int:
        return self.client.count(collection_name=self.collection, exact=True).count

    def health(self) -> dict:
        exists = self.client.collection_exists(self.collection)
        return {
            "collection": self.collection,
            "exists": exists,
            "points": self.count() if exists else 0,
        }


def build_filter(filters: dict | None) -> models.Filter | None:
    """Turn a flat ``{field: value | [values]}`` dict into a Qdrant filter."""
    if not filters:
        return None
    conditions = []
    for key, value in filters.items():
        match = (
            models.MatchAny(any=list(value))
            if isinstance(value, (list, tuple, set))
            else models.MatchValue(value=value)
        )
        conditions.append(models.FieldCondition(key=key, match=match))
    return models.Filter(must=conditions)
