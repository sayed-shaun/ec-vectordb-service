"""Orchestration: embed with every model, retrieve per model, fuse, respond."""

from __future__ import annotations

import logging
import time

from src.core.config import MODELS_BY_NAME, settings
from src.embedding import EmbedderRegistry, registry as default_registry
from src.fusion import reciprocal_rank_fusion, weighted_score_fusion
from src.schemas import (
    Document,
    Hit,
    ModelVote,
    QueryRequest,
    QueryResponse,
)
from src.store import VectorStore, build_filter

log = logging.getLogger(__name__)


class RetrievalService:
    def __init__(
        self, store: VectorStore, embedders: EmbedderRegistry | None = None
    ) -> None:
        self.store = store
        self.embedders = embedders or default_registry

    def ingest(self, documents: list[Document]) -> tuple[int, list[str]]:
        texts = [d.text for d in documents]
        vectors_by_model = self.embedders.embed_passages(texts)
        count = self.store.upsert(
            doc_ids=[d.doc_id for d in documents],
            texts=texts,
            vectors_by_model=vectors_by_model,
            metadatas=[d.metadata for d in documents],
        )
        return count, list(vectors_by_model)

    def query(self, req: QueryRequest) -> QueryResponse:
        started = time.perf_counter()
        model_names = req.models or self.embedders.names
        unknown = [m for m in model_names if m not in self.embedders.names]
        if unknown:
            raise ValueError(f"unknown model(s): {', '.join(unknown)}")

        per_model_limit = min(
            req.top_k * settings.candidate_multiplier, settings.max_candidates_per_model
        )

        query_vectors = self.embedders.embed_queries(req.question, models=model_names)
        candidates = self.store.search_all(
            vectors_by_model=query_vectors,
            limit=per_model_limit,
            query_filter=build_filter(req.filters),
        )

        weights = req.weights or {
            name: MODELS_BY_NAME[name].weight
            for name in model_names
            if name in MODELS_BY_NAME
        }
        fuse = reciprocal_rank_fusion if req.fusion == "rrf" else weighted_score_fusion
        fused = fuse(
            candidates,
            weights=weights,
            k=settings.rrf_k,
            top_k=req.top_k,
            min_votes=req.min_votes,
        )

        hits = []
        for f in fused:
            payload = dict(f.payload)
            text = payload.pop("text", None)
            payload.pop("doc_id", None)
            hits.append(
                Hit(
                    doc_id=f.doc_id,
                    score=round(f.fused_score, 6),
                    votes=f.votes,
                    per_model={
                        name: ModelVote(
                            rank=rank, score=round(f.per_model_score[name], 6)
                        )
                        for name, rank in f.per_model_rank.items()
                    },
                    text=text if req.include_text else None,
                    metadata=payload,
                )
            )

        return QueryResponse(
            question=req.question,
            fusion=req.fusion,
            models=model_names,
            took_ms=round((time.perf_counter() - started) * 1000, 2),
            hits=hits,
        )
