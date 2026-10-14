"""EC FAQ corpus loader: canonical questions and the tag each belongs to.

~18k questions over ~532 tags. Only the question is embedded; the tag rides
along in the payload and is returned with each hit.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
QUESTION_TAG_CSV = DATA_DIR / "question_tag.csv"


@dataclass(frozen=True)
class QuestionRow:
    qid: str
    question: str
    tag: str


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
