"""Reindex job status."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

__all__ = ["ReindexStatus"]


class ReindexStatus(BaseModel):
    status: Literal["idle", "running", "succeeded", "failed"] = "idle"
    phase: Literal["fetching", "indexing", "pruning"] | None = Field(
        None, description="Where a running job is."
    )
    total: int = Field(0, description="Questions in the fetched corpus.")
    indexed: int = Field(0, description="Questions embedded and upserted so far.")
    pruned: int = Field(0, description="Stale points removed once indexing finished.")
    started_at: float | None = Field(None, description="Unix time the job started.")
    finished_at: float | None = None
    error: str | None = None
