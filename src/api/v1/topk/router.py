"""Top-k retrieval endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.api.deps import get_topk_service
from src.api.v1.topk.schema import TopSimilarRequest, TopSimilarResponse
from src.topk_service import TopKService

router = APIRouter(prefix="/topk", tags=["topk"])


@router.post("/top_similar/", response_model=TopSimilarResponse)
def top_similar(
    req: TopSimilarRequest, ec: TopKService = Depends(get_topk_service)
) -> TopSimilarResponse:
    """Top-K similar canonical questions, each with its tag and cosine."""
    try:
        return ec.top_similar(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
