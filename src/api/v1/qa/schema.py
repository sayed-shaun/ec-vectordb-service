"""Request and response models for the QA endpoint."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["QaRequest", "QaResponse", "QaTurn"]


class QaTurn(BaseModel):
    """One exchange, as the caller replays it back to us."""

    question: str
    tag: str | None = None


class QaRequest(BaseModel):
    """Only ``question`` is required.

    ``history`` makes a follow-up answerable: "how much is the fee?" carries no
    subject on its own. The server stores nothing -- the caller holds the
    history and passes it back, so the service stays horizontally scalable.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"question": "আমার নাম ভুল আছে আমার nid কার্ডে কি করবো ?"}
        }
    )

    question: str = Field(..., min_length=1)
    history: list[QaTurn] = Field(
        default_factory=list,
        description="Prior turns, oldest first. Echoed back with this turn appended.",
    )
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
    history: list[QaTurn] = Field(
        default_factory=list, description="Pass this straight back on the next turn."
    )
    used_context: bool = Field(
        False, description="True when history rescued a question that failed standalone."
    )
    took_ms: float = 0.0
