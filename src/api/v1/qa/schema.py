"""Request and response models for the QA endpoint."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["QaRequest", "QaResponse", "QaTurn"]


class QaTurn(BaseModel):
    """One exchange, as the caller replays it back to us.

    ``question`` and ``answer`` are exactly what the rewrite service's own
    schema expects for a history entry -- this type exists to be that pair,
    nothing more.
    """

    question: str
    answer: str | None = None


class QaRequest(BaseModel):
    """Only ``question`` is required.

    A follow-up like "how much is the fee?" carries no subject on its own; an
    external service rewrites it into a standalone question given the
    conversation so far. Two ways to supply that conversation:

    ``chat_id`` -- the server holds the turns in memory, per process, for
    ``SESSION_TTL_S`` of inactivity (default 30 min). Simplest for a
    caller: send only ``question`` and the same ``chat_id`` each turn.

    ``history`` -- the caller holds every turn and echoes it back, exactly as
    ``QaResponse.history`` returns it. Slower to wire up, but stateless and
    horizontally scalable: no server memory, so any instance can serve any
    turn. Ignored when ``chat_id`` is given.

    Omit both for a single, context-free question.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"question": "আমার নাম ভুল আছে আমার nid কার্ডে কি করবো ?"}
        }
    )

    question: str = Field(..., min_length=1)
    chat_id: str | None = Field(
        None, description="Server-held conversation id. Takes priority over `history`."
    )
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
        False, description="True when the rewrite service returned a different question."
    )
    rewritten_question: str | None = Field(
        None, description="What the rewrite service turned the question into, if it ran."
    )
    chat_id: str | None = Field(
        None, description="Echoed back; pass it on the next turn to keep the conversation."
    )
    took_ms: float = 0.0
