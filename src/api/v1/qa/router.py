"""The QA endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.api.deps import get_ec_service
from src.api.v1.qa.schema import QaRequest, QaResponse
from src.ec_service import EcBotService

router = APIRouter(tags=["qa"])


@router.post("/qa", response_model=QaResponse)
def ask(req: QaRequest, ec: EcBotService = Depends(get_ec_service)) -> QaResponse:
    """Answer one question.

    Three models retrieve independently and vote on a tag. At least
    ``min_model_agreement`` of them must land on the same tag, otherwise the
    question is treated as unanswerable and the fallback copy is returned with
    ``confident=False``.
    """
    try:
        return ec.answer(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
