"""Operational response models.

`store` and `models` are open dicts on purpose: they are diagnostics, and
pinning their shape here would mean editing two files every time a new fact
about a model or the collection is worth reporting.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

__all__ = ["HealthResponse"]


class HealthResponse(BaseModel):
    status: str
    store: dict[str, Any] = Field(default_factory=dict)
    models: dict[str, Any] = Field(default_factory=dict)
    sessions: dict[str, Any] = Field(default_factory=dict)

