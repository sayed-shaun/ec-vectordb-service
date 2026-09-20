"""EC FAQ bot endpoints.

The paths are deliberately unversioned and unprefixed: they have to match the
existing service at `:8002` byte for byte. See src/api/v1/__init__.py.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.api.deps import get_ec_service
from src.api.v1.ec_bot.schema import TopSimilarRequest, TopSimilarResponse
from src.ec import AnswerStore, audit, load_questions
from src.ec_service import EcBotService

router = APIRouter(prefix="/ec_bot", tags=["ec_bot"])


@router.post("/top_similar/", response_model=TopSimilarResponse)
def top_similar(
    req: TopSimilarRequest, ec: EcBotService = Depends(get_ec_service)
) -> TopSimilarResponse:
    """Top-K similar canonical questions, with the voted tag and its answer.

    Request and the four documented keys of each `top_similar` entry match the
    existing EC bot service; the remaining fields are additive.
    """
    try:
        return ec.top_similar(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/audit")
def dataset_audit() -> dict:
    """Integrity report over question_tag.csv and tag_answer.json."""
    try:
        return audit(load_questions(), AnswerStore.load())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"dataset missing: {exc}") from exc
