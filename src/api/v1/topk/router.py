"""EC FAQ top-k endpoint.

The path is deliberately unversioned and unprefixed: it has to match the
existing service at `:8002` byte for byte. See src/api/v1/__init__.py.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.api.deps import get_ec_service
from src.api.v1.ec_bot.schema import TopSimilarRequest, TopSimilarResponse
from src.ec_service import EcBotService

router = APIRouter(prefix="/ec_bot", tags=["ec_bot"])


@router.post("/top_similar/", response_model=TopSimilarResponse)
def top_similar(
    req: TopSimilarRequest, ec: EcBotService = Depends(get_ec_service)
) -> TopSimilarResponse:
    """Top-K similar canonical questions, each with its tag and cosine."""
    try:
        return ec.top_similar(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
