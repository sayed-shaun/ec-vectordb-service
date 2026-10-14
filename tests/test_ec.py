import json
from pathlib import Path

import pytest

from src.api.v1.ec_bot.schema import TopSimilarRequest
from src.core.config import MODEL_SPECS
from src.ec import load_questions
from src.ec_service import EcBotService
from src.schemas import Document

CORPUS = [
    ("q1", "নতুন ভোটার হতে কি কি কাগজ লাগে", "new_voter_docs"),
    ("q2", "নতুন ভোটার হওয়ার জন্য কি কি লাগে", "new_voter_docs"),
    ("q3", "নতুন ভোটার নিবন্ধনে কোন কাগজ প্রয়োজন", "new_voter_docs"),
    ("q4", "আমার একাউন্ট লক হয়ে গেছে", "account_locked"),
    ("q5", "এনআইডি একাউন্ট আনলক করব কিভাবে", "account_locked"),
    ("q6", "সংশোধন ফি কত টাকা", "fee_amount"),
]


@pytest.fixture
def ec(service):
    service.ingest(
        [Document(doc_id=q, text=t, metadata={"tag": tag, "lang": "bn"}) for q, t, tag in CORPUS]
    )
    return EcBotService(service)


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


def test_response_has_no_answer_or_gate_fields(ec):
    body = json.loads(
        ec.top_similar(TopSimilarRequest(question="সংশোধন ফি", top_k=3)).model_dump_json()
    )
    assert not {"answer", "tag_votes", "confident", "fallback_reason"} & set(body)


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
    assert all(set(h.per_model) == {MODEL_SPECS[0].name} for h in res.top_similar)


@pytest.mark.skipif(
    not Path("data/question_tag.csv").exists(), reason="dataset not fetched"
)
def test_real_dataset_loads_every_row_with_a_tag():
    rows = load_questions()
    assert rows
    assert all(r.question and r.tag for r in rows)
