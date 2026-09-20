"""An in-memory, TTL-expiring store for follow-up conversation history.

Ephemeral by design: a chat's turns exist only long enough to answer its next
follow-up, and are forgotten once it goes idle. There is nothing here worth a
durable record -- a service restart forgetting every open conversation is an
acceptable cost for never writing to disk, and SQLite's actual advantage,
surviving a restart, is not a property this data should have.

Single-process only. This is a dict in process memory, not shared across
uvicorn workers or container replicas -- see the Dockerfile, which runs one
worker today. Scale this out and a chat's turn 2 can land on a different
worker than turn 1 and lose its history with no error, just silent amnesia.
The fix at that point is a shared store (Redis, or the SQLite file this
deliberately avoided), not more of this module.
"""

from __future__ import annotations

import threading
import time

from src.core.config import settings


class InMemoryStore:
    """Ephemeral, per-process conversation history, keyed by ``chat_id``.

    Turns are ``{"question", "answer"}`` dicts -- the exact shape
    ``src.rewrite`` already expects, so no translation happens at the
    boundary. Thread-safe: FastAPI serves sync routes from a thread pool, and
    every method takes the same lock.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # chat_id -> (last_touched, turns)
        self._sessions: dict[str, tuple[float, list[dict[str, str]]]] = {}

    def _sweep_locked(self) -> None:
        """Drop idle-past-TTL sessions, then oldest-first past the count cap.

        Must be called with ``self._lock`` held. The cap exists because
        ``chat_id`` is caller-supplied and this service has no
        authentication: without one, an attacker could grow this dict
        without bound between sweeps.
        """
        now = time.time()
        expired = [
            cid for cid, (touched, _) in self._sessions.items()
            if now - touched > settings.session_ttl_s
        ]
        for cid in expired:
            del self._sessions[cid]

        over = len(self._sessions) - settings.session_max_count
        if over > 0:
            for cid, _ in sorted(self._sessions.items(), key=lambda kv: kv[1][0])[:over]:
                del self._sessions[cid]

    def get(self, chat_id: str) -> list[dict[str, str]]:
        """This chat's turns so far, oldest first. Empty if unseen or expired."""
        with self._lock:
            self._sweep_locked()
            entry = self._sessions.get(chat_id)
            return list(entry[1]) if entry else []

    def append(self, chat_id: str, question: str, answer: str | None) -> None:
        with self._lock:
            self._sweep_locked()
            _, turns = self._sessions.get(chat_id, (0.0, []))
            turns.append({"question": question, "answer": answer or ""})
            self._sessions[chat_id] = (time.time(), turns)

    def clear(self, chat_id: str) -> None:
        """Forget a chat immediately, for a caller that knows it has ended."""
        with self._lock:
            self._sessions.pop(chat_id, None)

    def count(self) -> int:
        """Live session count, for /health -- not part of the request path."""
        with self._lock:
            return len(self._sessions)


# A process-wide singleton, matching the `registry = EmbedderRegistry()`
# pattern in src/embedding.py: one store per running instance.
store = InMemoryStore()
