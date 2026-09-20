"""Shared FastAPI dependencies.

The service objects are constructed once in the application lifespan and
handed here, so a router never imports `main` -- the app imports routers and
never the other way round, which is what keeps this package free of cycles.
"""

from __future__ import annotations

from fastapi import HTTPException

from src.ec_service import EcBotService
from src.service import RetrievalService

_service: RetrievalService | None = None
_ec_service: EcBotService | None = None


def set_services(
    retrieval: RetrievalService | None, ec_bot: EcBotService | None = None
) -> None:
    global _service, _ec_service
    _service = retrieval
    _ec_service = ec_bot


def get_service() -> RetrievalService:
    if _service is None:  # pragma: no cover - only if startup failed
        raise HTTPException(status_code=503, detail="service not ready")
    return _service


def get_ec_service() -> EcBotService:
    if _ec_service is None:
        raise HTTPException(
            status_code=503,
            detail="EC dataset not loaded; expected data/tag_answer.json",
        )
    return _ec_service
