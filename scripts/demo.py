"""Ingest a few Bangla passages with the real models and run a query.

    python scripts/demo.py "আপনার প্রশ্ন"

First run downloads ~3.5 GB of weights and takes several minutes.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from src.schemas import Document, QueryRequest  # noqa: E402
from src.service import RetrievalService  # noqa: E402
from src.store import VectorStore  # noqa: E402

SAMPLES = [
    ("d1", "ভোটার তালিকা হালনাগাদ কার্যক্রম আগামী মাসে শুরু হবে বলে জানিয়েছে নির্বাচন কমিশন।", "ec"),
    ("d2", "জাতীয় পরিচয়পত্র সংশোধনের আবেদন অনলাইনে করা যাবে।", "ec"),
    ("d3", "পদ্মা সেতু বাংলাদেশের দীর্ঘতম সেতু, যা ২০২২ সালে উদ্বোধন করা হয়।", "infra"),
    ("d4", "ঢাকা শহরের যানজট নিরসনে মেট্রোরেল চালু হয়েছে।", "infra"),
    ("d5", "নতুন ভোটার নিবন্ধনের জন্য জন্ম সনদ ও পিতামাতার এনআইডি প্রয়োজন।", "ec"),
]


def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else "নতুন ভোটার হতে কী কী কাগজ লাগে?"

    store = VectorStore()
    store.ensure_collection()
    service = RetrievalService(store)

    docs = [Document(doc_id=i, text=t, metadata={"source": s, "lang": "bn"}) for i, t, s in SAMPLES]
    count, models = service.ingest(docs)
    print(f"ingested {count} documents across {models}\n")

    res = service.query(QueryRequest(question=question, top_k=3))
    print(f"Q: {question}   ({res.took_ms} ms, fusion={res.fusion})\n")
    for i, hit in enumerate(res.hits, 1):
        ranks = ", ".join(f"{m}#{v.rank}({v.score:.3f})" for m, v in hit.per_model.items())
        print(f"{i}. [{hit.doc_id}] rrf={hit.score:.5f} votes={hit.votes}/{len(res.models)}")
        print(f"   {hit.text}")
        print(f"   {ranks}\n")


if __name__ == "__main__":
    main()
