"""EC FAQ corpus loader: canonical questions, their tags, and each tag's answer.

~18k questions over ~532 tags. Only the question is embedded; the tag rides
along in the payload and the answer is looked up from tag_answer.json when a
hit is returned.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
QUESTION_TAG_CSV = DATA_DIR / "question_tag.csv"
TAG_ANSWER_JSON = DATA_DIR / "tag_answer.json"


@dataclass(frozen=True)
class QuestionRow:
    qid: str
    question: str
    tag: str


class AnswerStore:
    """Tag -> answer lookup, loaded from tag_answer.json.

    ``reload`` swaps the whole mapping in one assignment, so a request in
    flight sees either the old answers or the new ones, never a mix.
    """

    def __init__(self, answers: dict[str, str] | None = None) -> None:
        self._answers = answers or {}

    @classmethod
    def load(cls, path: Path | str = TAG_ANSWER_JSON) -> "AnswerStore":
        store = cls()
        try:
            store.reload(path)
        except FileNotFoundError:
            log.warning("%s not found; hits will carry no answer until /reindex", path)
        return store

    def reload(self, path: Path | str = TAG_ANSWER_JSON) -> int:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError("tag_answer.json must map tag to answer")
        self._answers = data
        return len(data)

    def get(self, tag: str) -> str | None:
        return self._answers.get(tag)

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
