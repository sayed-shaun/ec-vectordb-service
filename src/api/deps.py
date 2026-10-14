"""Shared FastAPI dependencies.

The service objects are constructed once in the application lifespan and
handed here, so a router never imports `main` -- the app imports routers and
never the other way round, which is what keeps this package free of cycles.
"""

from __future__ import annotations

from fastapi import HTTPException

from src.dataset import fetch_answers, fetch_dataset
from src.reindex import Reindexer
from src.service import RetrievalService
from src.topk_service import TopKService

_service: RetrievalService | None = None
_topk_service: TopKService | None = None
_reindexer: Reindexer | None = None


def set_services(retrieval: RetrievalService | None) -> None:
    global _service, _topk_service, _reindexer
    _service = retrieval
    if retrieval is None:
        _topk_service = _reindexer = None
        return
    _topk_service = TopKService(retrieval)
    _reindexer = Reindexer(retrieval, fetch=fetch_dataset, fetch_answers=fetch_answers)


def get_service() -> RetrievalService:
    if _service is None:  # pragma: no cover - only if startup failed
        raise HTTPException(status_code=503, detail="service not ready")
    return _service


def get_topk_service() -> TopKService:
    if _topk_service is None:  # pragma: no cover - only if startup failed
        raise HTTPException(status_code=503, detail="service not ready")
    return _topk_service


def get_reindexer() -> Reindexer:
    if _reindexer is None:  # pragma: no cover - only if startup failed
        raise HTTPException(status_code=503, detail="service not ready")
    return _reindexer
