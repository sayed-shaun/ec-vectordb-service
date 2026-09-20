"""Query rewriting: an external service turns a follow-up into a standalone
question, so the retrieval pipeline never has to understand conversation.

This replaces a hand-built anchor heuristic that concatenated the last
self-sufficient question in front of a follow-up. That approach could only
ever produce "question1 question2" -- not a real question -- and had to guess,
from a boolean recorded per turn, which prior turn was worth anchoring on. An
external rewriter with actual language understanding does this properly:
"এনআইডি রি-ইস্যু করতে কত দিন সময় লাগবে?" is a real question; string
concatenation cannot produce one.

The call is best-effort. If the service is slow, unreachable, or answers with
something we cannot parse, the original question goes to retrieval unchanged
-- a rewrite failure must never be a request failure.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

log = logging.getLogger(__name__)


def rewrite_question(
    question: str,
    history: list[dict[str, str]],
    *,
    url: str,
    timeout: float,
) -> tuple[str, bool]:
    """Return ``(effective_question, was_rewritten)``.

    ``history`` is ``[{"question": ..., "answer": ...}, ...]``, oldest first --
    the shape the rewrite service's own schema documents. An empty ``url``
    disables the call entirely, so the feature is opt-in and local development
    or tests never need the service reachable.
    """
    if not url or not history:
        return question, False

    payload = json.dumps({"question": question, "history": history}, ensure_ascii=False)
    request = urllib.request.Request(
        url,
        data=payload.encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        log.warning("rewrite service unavailable (%s); using the question as asked", exc)
        return question, False

    rewritten = body.get("rewritten_question")
    if not isinstance(rewritten, str) or not rewritten.strip():
        return question, False
    return rewritten, bool(body.get("was_rewritten"))
