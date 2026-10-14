"""Reindex trigger and status."""

from __future__ import annotations

import hmac

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException

from src.api.deps import get_reindexer
from src.api.v1.reindex.schema import ReindexStatus
from src.core.config import settings
from src.reindex import Reindexer


def require_token(x_reindex_token: str | None = Header(None)) -> None:
    expected = settings.reindex_token
    if expected and not hmac.compare_digest(x_reindex_token or "", expected):
        raise HTTPException(status_code=401, detail="invalid or missing X-Reindex-Token")


router = APIRouter(
    prefix="/reindex", tags=["reindex"], dependencies=[Depends(require_token)]
)


@router.post("", response_model=ReindexStatus, status_code=202)
def start_reindex(
    background: BackgroundTasks, job: Reindexer = Depends(get_reindexer)
) -> ReindexStatus:
    """Fetch the corpus from GitHub and re-index it in the background."""
    if not job.begin():
        raise HTTPException(status_code=409, detail="a reindex is already running")
    background.add_task(job.run)
    return job.status()


@router.get("", response_model=ReindexStatus)
def reindex_status(job: Reindexer = Depends(get_reindexer)) -> ReindexStatus:
    return job.status()
