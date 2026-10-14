"""EC FAQ top-k retrieval: normalise, retrieve, shape the hits."""

from __future__ import annotations

import logging
import time

from src.api.v1.ec_bot.schema import (
    SimilarQuestion,
    TopSimilarRequest,
    TopSimilarResponse,
)
from src.normalize import normalize
from src.schemas import QueryRequest
from src.service import RetrievalService

log = logging.getLogger(__name__)


class EcBotService:
    def __init__(self, retrieval: RetrievalService) -> None:
        self.retrieval = retrieval

    def top_similar(self, req: TopSimilarRequest) -> TopSimilarResponse:
        started = time.perf_counter()
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
            similar.append(
                SimilarQuestion(
                    rank=rank,
                    question=hit.text or "",
                    tag=hit.metadata.get("tag", ""),
                    cosine_similarity=round(sum(cosines) / len(cosines), 6) if cosines else 0.0,
                    votes=hit.votes,
                    per_model=dict(hit.per_model),
                )
            )

        return TopSimilarResponse(
            input_question=req.question,
            normalized_question=normalized,
            top_similar=similar,
            took_ms=round((time.perf_counter() - started) * 1000, 2),
        )
