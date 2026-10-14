"""Reindex: fetch the corpus from GitHub, embed it, prune stale points.

The router claims the job with ``begin()`` and hands ``run()`` to FastAPI's
BackgroundTasks, so it runs after the 202 has been sent. One job at a time, held in this process's memory -- a second worker or replica
would not see it, the same trade-off the old session store made. Indexing
upserts in place, so queries keep answering from the previous index until the
new rows overwrite it; pruning runs last, and only after every row landed.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

from src.api.v1.reindex.schema import ReindexStatus
from src.core.config import settings
from src.dataset import fetch_dataset
from src.ec import QUESTION_TAG_CSV, TAG_ANSWER_JSON, AnswerStore, load_questions
from src.indexer import index_questions
from src.service import RetrievalService

log = logging.getLogger(__name__)


class Reindexer:
    def __init__(
        self,
        service: RetrievalService,
        fetch: Callable[[Path, str], int] = fetch_dataset,
        dest: Path = QUESTION_TAG_CSV,
        fetch_answers: Callable[[Path, str], int] | None = None,
        answers_dest: Path = TAG_ANSWER_JSON,
    ) -> None:
        self.service = service
        self._fetch = fetch
        self._dest = dest
        self._fetch_answers = fetch_answers
        self._answers_dest = answers_dest
        self._lock = threading.Lock()
        self._state = ReindexStatus()

    def status(self) -> ReindexStatus:
        with self._lock:
            return self._state.model_copy()

    def begin(self) -> bool:
        """Claim the job; False when one is already running."""
        with self._lock:
            if self._state.status == "running":
                return False
            self._state = ReindexStatus(
                status="running", phase="fetching", started_at=time.time()
            )
            return True

    def _set(self, **fields) -> None:
        with self._lock:
            self._state = self._state.model_copy(update=fields)

    def run(self, token: str) -> None:
        """The job body. Never raises: a failure lands in the status.

        ``token`` is the caller's GitHub token, used for the download and then
        dropped; it is not kept on the instance or reported in the status.
        """
        try:
            self._set(phase="fetching")
            self._fetch(self._dest, token)
            if self._fetch_answers:
                self._fetch_answers(self._answers_dest, token)
            answers = AnswerStore.load(self._answers_dest)
            questions = load_questions(self._dest)
            if not questions:
                raise ValueError("dataset has no usable rows")

            self._set(phase="indexing", total=len(questions), indexed=0)
            index_questions(
                self.service,
                questions,
                settings.INDEX_NORMALIZER,
                settings.INDEX_BATCH_SIZE,
                on_progress=lambda n: self._set(indexed=n),
                answers=answers,
            )

            self._set(phase="pruning")
            pruned = self.service.store.prune({q.qid for q in questions})
            self._set(status="succeeded", phase=None, pruned=pruned, finished_at=time.time())
            log.info("reindex done: %d indexed, %d pruned", len(questions), pruned)
        except Exception as exc:
            log.exception("reindex failed")
            self._set(status="failed", phase=None, error=str(exc), finished_at=time.time())
