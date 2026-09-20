"""EC FAQ bot request/response models.

Field names and ordering mirror the existing service at `:8002` so this can
stand in for it. Everything beyond `top_similar`'s four documented keys is
additive, which existing consumers ignore.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.schemas import ModelVote

__all__ = [
    "ModelVote",
    "SimilarQuestion",
    "TagVote",
    "TopSimilarRequest",
    "TopSimilarResponse",
]


class TopSimilarRequest(BaseModel):
    """Only ``question`` is required; omit ``models`` to use all three."""

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
        None, description="Subset of e5, qwen3, bge_m3. Omit for all three."
    )
    fusion: Literal["rrf", "weighted_score"] = "rrf"
    min_votes: int = Field(1, ge=1, le=3)
    min_cosine: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "The winning tag's best hit must reach this cosine, else the "
            "fallback is returned. Defaults to MIN_COSINE (0.70)."
        ),
    )
    min_model_agreement: int | None = Field(
        None,
        ge=1,
        le=3,
        description=(
            "How many models must independently pick the winning tag. Below "
            "this the fallback tag is returned instead. Defaults to "
            "MIN_MODEL_AGREEMENT (2)."
        ),
    )
    normalizer: Literal["compat", "safe", "none"] = "compat"
    include_answer: bool = True


class SimilarQuestion(BaseModel):
    rank: int
    question: str
    tag: str
    cosine_similarity: float
    votes: int = 0
    per_model: dict[str, ModelVote] = Field(default_factory=dict)


class TagVote(BaseModel):
    tag: str
    score: float
    hits: int
    best_rank: int
    answer: str | None = None


class TopSimilarResponse(BaseModel):
    input_question: str
    normalized_question: str
    top_similar: list[SimilarQuestion]
    tag: str | None = None
    answer: str | None = None
    tag_votes: list[TagVote] = Field(default_factory=list)
    took_ms: float = 0.0
    per_model_tag: dict[str, str] = Field(
        default_factory=dict, description="The tag each model picked on its own."
    )
    model_agreement: int = Field(0, description="How many models backed the returned tag.")
    top_cosine: float = Field(0.0, description="Best cosine among the winning tag's hits.")
    fallback_reason: str | None = Field(
        None, description="no_match, low_similarity or model_disagreement."
    )
    confident: bool = Field(
        True, description="False when agreement fell short and the fallback was returned."
    )
