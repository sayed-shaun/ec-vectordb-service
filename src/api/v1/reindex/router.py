"""Reindex trigger and status."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException

from src.api.deps import get_reindexer
from src.api.v1.reindex.schema import ReindexStatus
from src.reindex import Reindexer


def github_token(x_github_token: str | None = Header(None)) -> str:
    """The caller's GitHub token, used once for the download.

    A header, not the body or query string, which proxies and access logs
    record. The token is the credential: a caller without read access to the
    dataset repo gets a 401 from GitHub and nothing is indexed.
    """
    token = (x_github_token or "").strip()
    if not token:
        raise HTTPException(status_code=401, detail="X-GitHub-Token header is required")
    return token


router = APIRouter(prefix="/reindex", tags=["reindex"])


@router.post("", response_model=ReindexStatus, status_code=202)
def start_reindex(
    background: BackgroundTasks,
    token: str = Depends(github_token),
    job: Reindexer = Depends(get_reindexer),
) -> ReindexStatus:
    """Fetch the corpus from GitHub with the caller's token and re-index it."""
    if not job.begin():
        raise HTTPException(status_code=409, detail="a reindex is already running")
    background.add_task(job.run, token)
    return job.status()


@router.get("", response_model=ReindexStatus)
def reindex_status(job: Reindexer = Depends(get_reindexer)) -> ReindexStatus:
    return job.status()
