"""Index the EC FAQ corpus into Qdrant.

    python scripts/ingest_ec.py [--limit N] [--batch N] [--normalizer compat]

Embeds ~18k canonical questions with every configured model. Only the question text
is embedded; the tag and the original surface form ride along in the payload
and come back with each hit.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

sys.path.insert(0, ".")

from src.core.config import MODEL_SPECS, settings  # noqa: E402
from src.core.logger import setup_logging  # noqa: E402
from src.ec import load_questions  # noqa: E402
from src.indexer import index_questions  # noqa: E402
from src.service import RetrievalService  # noqa: E402
from src.store import VectorStore  # noqa: E402

setup_logging()
log = logging.getLogger("ingest")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="index only the first N rows")
    ap.add_argument("--batch", type=int, default=settings.INDEX_BATCH_SIZE, help="rows per upsert")
    ap.add_argument("--normalizer", default=settings.INDEX_NORMALIZER, choices=["compat", "safe", "none"])
    ap.add_argument("--recreate", action="store_true", help="drop the collection first")
    ap.add_argument(
        "--resume", action="store_true",
        help="skip questions already indexed (safe to re-run after a crash)",
    )
    args = ap.parse_args()

    questions = load_questions()
    log.info("corpus: %d questions, %d tags", len(questions), len({q.tag for q in questions}))

    if args.limit:
        questions = questions[: args.limit]

    store = VectorStore()
    if args.recreate and store.client.collection_exists(store.collection):
        log.info("dropping collection %s", store.collection)
        store.client.delete_collection(store.collection)
    store.ensure_collection()
    service = RetrievalService(store)

    for spec in MODEL_SPECS:
        log.info(
            "model %s -> %s/%s", spec.name, spec.resolved_device(), spec.resolved_dtype()
        )

    if args.resume and not args.recreate:
        already = store.existing_doc_ids([q.qid for q in questions])
        if already:
            before = len(questions)
            questions = [q for q in questions if q.qid not in already]
            log.info("resume: %d already indexed, %d to go", before - len(questions), len(questions))
    if not questions:
        log.info("nothing to do; collection holds %d points", store.count())
        return

    started = time.perf_counter()
    index_questions(service, questions, args.normalizer, args.batch)

    log.info("done: %d points in %.1f min", store.count(), (time.perf_counter() - started) / 60)


if __name__ == "__main__":
    main()
