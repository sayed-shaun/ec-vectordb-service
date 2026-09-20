"""EC FAQ domain layer: questions -> tag -> answer.

~18k canonical questions over ~532 tags. Only questions are embedded; the
answer is looked up from the winning tag. See the README for why.
"""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
QUESTION_TAG_CSV = DATA_DIR / "question_tag.csv"
TAG_ANSWER_JSON = DATA_DIR / "tag_answer.json"

FALLBACK_TAG = "unable_to_answer"


@dataclass(frozen=True)
class QuestionRow:
    qid: str
    question: str
    tag: str


class AnswerStore:
    """Tag -> answer lookup, loaded from tag_answer.json."""

    def __init__(self, answers: dict[str, str]) -> None:
        self._answers = answers

    @classmethod
    def load(cls, path: Path | str = TAG_ANSWER_JSON) -> "AnswerStore":
        with open(path, encoding="utf-8") as fh:
            return cls(json.load(fh))

    def get(self, tag: str) -> str | None:
        return self._answers.get(tag)

    def fallback(self) -> str | None:
        """The official "we cannot answer this" copy."""
        return self._answers.get(FALLBACK_TAG)

    def __contains__(self, tag: str) -> bool:
        return tag in self._answers

    def __len__(self) -> int:
        return len(self._answers)


def load_questions(path: Path | str = QUESTION_TAG_CSV) -> list[QuestionRow]:
    """Read question_tag.csv.

    UTF-8 with a BOM, hence ``utf-8-sig``: plain ``utf-8`` glues a zero-width
    no-break space to the first header and loses the ``question`` column.
    """
    rows: list[QuestionRow] = []
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            question = (row.get("question") or "").strip()
            tag = (row.get("tag") or "").strip()
            if not question or not tag:
                log.warning("skipping row %d: missing question or tag", i)
                continue
            rows.append(QuestionRow(qid=f"q{i:06d}", question=question, tag=tag))
    return rows


def audit(questions: list[QuestionRow], answers: AnswerStore) -> dict:
    """Integrity report over the dataset pair. Run it before every reindex."""
    by_tag: dict[str, int] = defaultdict(int)
    for q in questions:
        by_tag[q.tag] += 1

    missing = sorted(t for t in by_tag if t not in answers)
    degenerate = sorted(
        (
            (t, answers.get(t) or "", by_tag[t])
            for t in by_tag
            if t in answers and len((answers.get(t) or "").strip()) < 25
        ),
        key=lambda row: -row[2],
    )
    return {
        "questions": len(questions),
        "reachable_tags": len(by_tag),
        "answer_entries": len(answers),
        "tags_without_answer": missing,
        "unreachable_answer_tags": len(answers) - len(by_tag),
        "degenerate_answers": [
            {"tag": t, "answer": a, "questions": n} for t, a, n in degenerate
        ],
        "questions_with_degenerate_answer": sum(n for _, _, n in degenerate),
    }
