"""Internal DTOs for the retrieval pipeline.

These are not an HTTP contract -- no route exposes them. They are the types
``RetrievalService`` speaks in, reused by the EC layer and the ingest script.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

__all__ = ["Document", "Hit", "ModelVote", "QueryRequest", "QueryResponse"]


class ModelVote(BaseModel):
    """One model's opinion of one hit: where it ranked it, and how strongly."""

    rank: int
    score: float


class Document(BaseModel):
    doc_id: str = Field(..., description="Caller-owned id; re-ingesting it overwrites in place.")
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(10, ge=1, le=100)
    models: list[str] | None = None
    fusion: Literal["rrf", "weighted_score"] = "rrf"
    min_votes: int = Field(1, ge=1, le=3)
    weights: dict[str, float] | None = None
    filters: dict[str, Any] | None = None
    include_text: bool = True


class Hit(BaseModel):
    doc_id: str
    score: float
    votes: int
    per_model: dict[str, ModelVote]
    text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    question: str
    fusion: str
    models: list[str]
    took_ms: float
    hits: list[Hit]
