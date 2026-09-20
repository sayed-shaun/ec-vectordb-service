"""Request and response models for the QA endpoint."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["QaRequest", "QaResponse"]


class QaRequest(BaseModel):
    """Only ``question`` is required."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"question": "আমার নাম ভুল আছে আমার nid কার্ডে কি করবো ?"}
        }
    )

    question: str = Field(..., min_length=1)
    min_model_agreement: int | None = Field(
        None,
        ge=1,
        le=3,
        description=(
            "How many of the three models must independently pick the same tag. "
            "Below this the fallback answer is returned. Defaults to 2."
        ),
    )
    min_cosine: float | None = Field(
        None, ge=0.0, le=1.0, description="Similarity floor. Defaults to 0.70."
    )
    top_k: int = Field(10, ge=1, le=100, description="Retrieval depth behind the answer.")
    normalizer: str = Field("compat", description="compat, safe or none.")


class QaResponse(BaseModel):
    """A caller that only wants the answer can read ``answer`` and stop.

    ``confident`` says whether the models agreed. When it is False the answer
    is the fallback copy, not a retrieved one, and a caller that escalates to
    a human should branch on it rather than on the answer text.
    """

    question: str
    answer: str | None
    tag: str | None
    confident: bool
    model_agreement: int = Field(0, description="How many models backed the tag.")
    per_model_tag: dict[str, str] = Field(default_factory=dict)
    top_cosine: float = Field(0.0, description="How well the question matched at all.")
    fallback_reason: str | None = Field(
        None, description="no_match, low_similarity or model_disagreement."
    )
    matched_question: str | None = Field(
        None, description="The closest canonical question, for logging and review."
    )
    took_ms: float = 0.0
