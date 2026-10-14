"""Embed and upsert the corpus; shared by scripts/ingest_ec.py and /reindex."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from src.ec import AnswerStore, QuestionRow
from src.normalize import normalize
from src.schemas import Document
from src.service import RetrievalService

log = logging.getLogger(__name__)


def index_questions(
    service: RetrievalService,
    questions: list[QuestionRow],
    normalizer: str,
    batch_size: int,
    on_progress: Callable[[int], None] | None = None,
    answers: AnswerStore | None = None,
) -> int:
    """Upsert ``questions`` in batches and return how many were written.

    Each point's payload carries its tag and that tag's answer.
    """
    total, started = 0, time.perf_counter()
    for i in range(0, len(questions), batch_size):
        chunk = questions[i : i + batch_size]
        docs = [
            Document(
                doc_id=q.qid,
                text=normalize(q.question, normalizer),
                metadata={
                    "tag": q.tag,
                    "answer": answers.get(q.tag) if answers else None,
                    "raw_question": q.question,
                    "source": "question_tag.csv",
                    "lang": "bn",
                },
            )
            for q in chunk
        ]
        count, _ = service.ingest(docs)
        total += count
        if on_progress:
            on_progress(total)
        elapsed = time.perf_counter() - started
        rate = total / elapsed if elapsed else 0
        remaining = (len(questions) - total) / rate if rate else 0
        log.info(
            "%d/%d  %.1f rows/s  eta %.1f min",
            total, len(questions), rate, remaining / 60,
        )
    return total
