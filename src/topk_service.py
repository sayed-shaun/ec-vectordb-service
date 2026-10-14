"""Top-k retrieval: normalise, retrieve, shape the hits (tag and answer ride in the payload)."""

from __future__ import annotations

import logging

from src.api.v1.topk.schema import (
    SimilarQuestion,
    TopSimilarRequest,
    TopSimilarResponse,
)
from src.normalize import normalize
from src.schemas import QueryRequest
from src.service import RetrievalService

log = logging.getLogger(__name__)


class TopKService:
    def __init__(self, retrieval: RetrievalService) -> None:
        self.retrieval = retrieval

    def top_similar(self, req: TopSimilarRequest) -> TopSimilarResponse:
        normalized = normalize(req.question, req.normalizer)

        inner = self.retrieval.query(
            QueryRequest(
                question=normalized,
                top_k=req.top_k,
                models=req.models,
                fusion=req.fusion,
                min_votes=req.min_votes,
                include_text=True,
            )
        )

        similar = []
        for rank, hit in enumerate(inner.hits, 1):
            cosines = [v.score for v in hit.per_model.values()]
            tag = hit.metadata.get("tag", "")
            similar.append(
                SimilarQuestion(
                    rank=rank,
                    question=hit.text or "",
                    answer=hit.metadata.get("answer"),
                    tag=tag,
                    cosine_similarity=round(sum(cosines) / len(cosines), 6) if cosines else 0.0,
                )
            )

        return TopSimilarResponse(
            input_question=req.question,
            normalized_question=normalized,
            top_similar=similar,
        )
