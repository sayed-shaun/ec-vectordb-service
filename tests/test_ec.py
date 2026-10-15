import json
from pathlib import Path

import pytest

from src.api.v1.topk.schema import TopSimilarRequest
from src.core.config import MODEL_SPECS
from src.ec import AnswerStore, load_questions
from src.topk_service import TopKService
from src.schemas import Document

CORPUS = [
    ("q1", "নতুন ভোটার হতে কি কি কাগজ লাগে", "new_voter_docs"),
    ("q2", "নতুন ভোটার হওয়ার জন্য কি কি লাগে", "new_voter_docs"),
    ("q3", "নতুন ভোটার নিবন্ধনে কোন কাগজ প্রয়োজন", "new_voter_docs"),
    ("q4", "আমার একাউন্ট লক হয়ে গেছে", "account_locked"),
    ("q5", "এনআইডি একাউন্ট আনলক করব কিভাবে", "account_locked"),
    ("q6", "সংশোধন ফি কত টাকা", "fee_amount"),
]

ANSWERS = {
    "new_voter_docs": "জন্ম সনদ ও পিতামাতার এনআইডি প্রয়োজন।",
    "account_locked": "আট ঘন্টা পর সঠিক তথ্য দিয়ে চেষ্টা করুন।",
    "fee_amount": "সংশোধন ফি ২৩০ টাকা।",
}


@pytest.fixture
def ec(service):
    service.ingest(
        [
            Document(doc_id=q, text=t, metadata={"tag": tag, "answer": ANSWERS.get(tag), "lang": "bn"})
            for q, t, tag in CORPUS
        ]
    )
    return TopKService(service)


def test_response_matches_reference_contract(ec):
    res = ec.top_similar(TopSimilarRequest(question="নতুন ভোটার হতে কি কি লাগে?", top_k=3))
    body = json.loads(res.model_dump_json())

    assert set(body) >= {"input_question", "normalized_question", "top_similar"}
    for entry in body["top_similar"]:
        assert set(entry) >= {"rank", "question", "tag", "cosine_similarity"}
        assert isinstance(entry["rank"], int)
        assert 0.0 <= entry["cosine_similarity"] <= 1.0

    assert body["input_question"] == "নতুন ভোটার হতে কি কি লাগে?"
    assert body["normalized_question"] == "নতুন ভোটার হতে কি কি লাগে"


def test_ranks_are_dense_and_one_based(ec):
    res = ec.top_similar(TopSimilarRequest(question="একাউন্ট লক", top_k=4))
    assert [h.rank for h in res.top_similar] == list(range(1, len(res.top_similar) + 1))


def test_top_k_bounds_the_result_count(ec):
    res = ec.top_similar(TopSimilarRequest(question="নতুন ভোটার", top_k=2))
    assert len(res.top_similar) == 2


def test_hits_are_ordered_best_first(ec):
    res = ec.top_similar(TopSimilarRequest(question="নতুন ভোটার হতে কি কি লাগে", top_k=5))
    cosines = [h.cosine_similarity for h in res.top_similar]
    assert cosines == sorted(cosines, reverse=True)


def test_each_hit_carries_its_tag(ec):
    res = ec.top_similar(TopSimilarRequest(question="নতুন ভোটার হতে কি কি লাগে", top_k=1))
    assert res.top_similar[0].tag == "new_voter_docs"


def test_each_hit_carries_its_tags_answer(ec):
    res = ec.top_similar(TopSimilarRequest(question="নতুন ভোটার হতে কি কি লাগে", top_k=6))
    assert res.top_similar
    for hit in res.top_similar:
        assert hit.answer == ANSWERS[hit.tag]


def test_a_tag_without_an_answer_gives_a_null_answer(service):
    service.ingest([Document(doc_id="x", text="নতুন প্রশ্ন", metadata={"tag": "orphan", "answer": None})])
    res = TopKService(service).top_similar(
        TopSimilarRequest(question="নতুন প্রশ্ন", top_k=1)
    )
    assert res.top_similar[0].tag == "orphan"
    assert res.top_similar[0].answer is None


def test_response_shape_is_exactly_question_hits_with_answer_and_tag(ec):
    body = json.loads(
        ec.top_similar(TopSimilarRequest(question="সংশোধন ফি", top_k=2)).model_dump_json()
    )
    assert list(body) == ["input_question", "normalized_question", "top_similar"]
    for hit in body["top_similar"]:
        assert list(hit) == ["rank", "question", "answer", "tag", "cosine_similarity"]


def test_answers_are_not_voted_on_or_gated(ec):
    """Answers ride on each hit; there is no winner, floor or fallback."""
    body = json.loads(
        ec.top_similar(TopSimilarRequest(question="সংশোধন ফি", top_k=3)).model_dump_json()
    )
    assert not {"answer", "tag", "tag_votes", "confident", "fallback_reason"} & set(body)


def test_normalizer_choice_is_reflected(ec):
    res = ec.top_similar(
        TopSimilarRequest(question="একাউন্ট, লক?", top_k=2, normalizer="none")
    )
    assert res.normalized_question == "একাউন্ট, লক?"


def test_a_named_model_subset_is_served(ec):
    res = ec.top_similar(
        TopSimilarRequest(
            question="নতুন ভোটার হতে কি কি লাগে", top_k=5, models=[MODEL_SPECS[0].name]
        )
    )
    assert res.top_similar


@pytest.mark.skipif(
    not Path("data/question_tag.csv").exists(), reason="dataset not fetched"
)
def test_real_dataset_loads_every_row_with_a_tag():
    rows = load_questions()
    assert rows
    assert all(r.question and r.tag for r in rows)


def test_answer_store_loads_the_tag_to_answer_map(tmp_path):
    path = tmp_path / "tag_answer.json"
    path.write_text('{"a": "one"}', encoding="utf-8")
    store = AnswerStore.load(path)
    assert store.get("a") == "one" and store.get("b") is None


def test_a_missing_answer_file_loads_empty(tmp_path):
    store = AnswerStore.load(tmp_path / "absent.json")
    assert len(store) == 0 and store.get("a") is None


def test_excluded_tags_are_not_loaded(tmp_path, monkeypatch):
    from src.core.config import settings

    path = tmp_path / "q.csv"
    path.write_text("question,tag\nq one,keep\nq two,drop\nq three,keep\n", encoding="utf-8")
    monkeypatch.setattr(settings, "EXCLUDED_TAGS", ["drop"])
    rows = load_questions(path)
    assert [r.tag for r in rows] == ["keep", "keep"]
    monkeypatch.setattr(settings, "EXCLUDED_TAGS", [])
    assert len(load_questions(path)) == 3
