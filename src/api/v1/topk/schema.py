"""EC FAQ top-k request/response models.

Field names and ordering mirror the existing service at `:8002` so this can
stand in for it.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.schemas import ModelVote

__all__ = ["ModelVote", "SimilarQuestion", "TopSimilarRequest", "TopSimilarResponse"]


class TopSimilarRequest(BaseModel):
    """Only ``question`` is required; omit ``models`` to use every configured one."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "আমার নাম ভুল আছে আমার nid কার্ডে কি করবো ?",
                "top_k": 10,
            }
        }
    )

    question: str = Field(..., min_length=1)
    top_k: int = Field(10, ge=1, le=100)
    models: list[str] | None = Field(
        None, description="Subset of the configured models (e5). Omit for all."
    )
    fusion: Literal["rrf", "weighted_score"] = "rrf"
    min_votes: int = Field(1, ge=1, le=3)
    normalizer: Literal["compat", "safe", "none"] = "compat"


class SimilarQuestion(BaseModel):
    rank: int
    question: str
    tag: str
    cosine_similarity: float
    votes: int = 0
    per_model: dict[str, ModelVote] = Field(default_factory=dict)


class TopSimilarResponse(BaseModel):
    input_question: str
    normalized_question: str
    top_similar: list[SimilarQuestion]
    took_ms: float = 0.0
