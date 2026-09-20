"""Health and warmup."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.deps import get_service
from src.api.v1.system.schema import HealthResponse
from src.core.memory import store as session_store
from src.embedding import registry
from src.service import RetrievalService

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health(service: RetrievalService = Depends(get_service)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        store=service.store.health(),
        models=registry.status(),
        sessions={"active": session_store.count()},
    )

