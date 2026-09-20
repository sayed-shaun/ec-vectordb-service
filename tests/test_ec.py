import json

import pytest

from src.ec import AnswerStore, audit, load_questions
from src.ec_service import EcBotService
from src.api.v1.ec_bot.schema import TopSimilarRequest
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
    "unable_to_answer": "দুঃখিত, ১০৫-এ কল করুন।",
}


@pytest.fixture
def ec(service):
    service.ingest(
        [Document(doc_id=q, text=t, metadata={"tag": tag, "lang": "bn"}) for q, t, tag in CORPUS]
    )
    return EcBotService(service, AnswerStore(ANSWERS))


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


def test_answer_resolved_from_voted_tag(ec):
    res = ec.top_similar(TopSimilarRequest(question="নতুন ভোটার হতে কি কি লাগে", top_k=5, min_cosine=0.0))
    assert res.tag == "new_voter_docs"
    assert res.answer == ANSWERS["new_voter_docs"]


def test_tag_voting_aggregates_multiple_questions(ec):
    res = ec.top_similar(TopSimilarRequest(question="নতুন ভোটার কাগজ", top_k=6))
    winner = res.tag_votes[0]
    assert winner.tag == "new_voter_docs"
    assert winner.hits >= 2
    assert winner.score >= res.tag_votes[1].score


def test_tag_votes_carry_their_answers(ec):
    res = ec.top_similar(TopSimilarRequest(question="সংশোধন ফি", top_k=6))
    for vote in res.tag_votes:
        assert vote.answer == ANSWERS[vote.tag]


def test_include_answer_false_omits_answer(ec):
    res = ec.top_similar(
        TopSimilarRequest(question="একাউন্ট লক", top_k=3, include_answer=False, min_cosine=0.0)
    )
    assert res.answer is None
    assert res.tag is not None


def test_normalizer_choice_is_reflected(ec):
    res = ec.top_similar(
        TopSimilarRequest(question="একাউন্ট, লক?", top_k=2, normalizer="none")
    )
    assert res.normalized_question == "একাউন্ট, লক?"


def test_audit_flags_missing_and_degenerate_answers():
    questions = [
        type("Q", (), {"qid": "q1", "question": "a", "tag": "good"})(),
        type("Q", (), {"qid": "q2", "question": "b", "tag": "stub"})(),
        type("Q", (), {"qid": "q3", "question": "c", "tag": "orphan"})(),
    ]
    answers = AnswerStore({"good": "একটি যথেষ্ট দীর্ঘ এবং অর্থবহ উত্তর এখানে আছে।", "stub": "tag"})
    report = audit(questions, answers)

    assert report["tags_without_answer"] == ["orphan"]
    assert [d["tag"] for d in report["degenerate_answers"]] == ["stub"]
    assert report["questions_with_degenerate_answer"] == 1


def test_degenerate_answers_are_ordered_by_blast_radius():
    """The head of the list is reported as 'worst', so it must be the biggest."""
    questions = (
        [type("Q", (), {"qid": f"s{i}", "question": "x", "tag": "small"})() for i in range(2)]
        + [type("Q", (), {"qid": f"b{i}", "question": "y", "tag": "big"})() for i in range(50)]
    )
    answers = AnswerStore({"small": "tag", "big": "tag"})
    report = audit(questions, answers)
    assert [d["tag"] for d in report["degenerate_answers"]] == ["big", "small"]


@pytest.mark.skipif(
    not (__import__("pathlib").Path("data/question_tag.csv").exists()),
    reason="dataset not fetched",
)
def test_real_dataset_has_full_answer_coverage():
    report = audit(load_questions(), AnswerStore.load())
    assert report["questions"] == 17995
    assert report["reachable_tags"] == 532
    assert report["tags_without_answer"] == []


def test_agreeing_models_return_the_voted_answer(ec):
    res = ec.top_similar(TopSimilarRequest(question="নতুন ভোটার হতে কি কি লাগে", top_k=5, min_cosine=0.0))
    assert res.confident is True
    assert res.model_agreement >= 2
    assert res.tag == "new_voter_docs"
    assert res.answer == ANSWERS["new_voter_docs"]


def test_per_model_tag_reports_each_model_separately(ec):
    res = ec.top_similar(TopSimilarRequest(question="একাউন্ট লক", top_k=5, min_cosine=0.0))
    assert set(res.per_model_tag) == {"e5", "qwen3", "bge_m3"}
    assert all(t in ANSWERS for t in res.per_model_tag.values())


def test_three_way_disagreement_falls_back(ec, monkeypatch):
    """All three models on different tags means the question is unanswerable."""
    monkeypatch.setattr(
        EcBotService,
        "_per_model_tags",
        lambda self, similar: {
            "e5": "new_voter_docs",
            "qwen3": "account_locked",
            "bge_m3": "fee_amount",
        },
    )
    res = ec.top_similar(TopSimilarRequest(question="নতুন ভোটার হতে কি কি লাগে", top_k=5, min_cosine=0.0))
    assert res.confident is False
    assert res.model_agreement < 2
    assert res.tag == "unable_to_answer"
    assert res.answer == ANSWERS["unable_to_answer"]


def test_two_of_three_is_enough(ec, monkeypatch):
    monkeypatch.setattr(
        EcBotService,
        "_per_model_tags",
        lambda self, similar: {
            "e5": "new_voter_docs",
            "qwen3": "new_voter_docs",
            "bge_m3": "fee_amount",
        },
    )
    res = ec.top_similar(TopSimilarRequest(question="নতুন ভোটার হতে কি কি লাগে", top_k=5, min_cosine=0.0))
    assert res.confident is True
    assert res.model_agreement == 2
    assert res.tag == "new_voter_docs"


def test_agreement_threshold_is_overridable(ec, monkeypatch):
    monkeypatch.setattr(
        EcBotService,
        "_per_model_tags",
        lambda self, similar: {
            "e5": "new_voter_docs",
            "qwen3": "new_voter_docs",
            "bge_m3": "fee_amount",
        },
    )
    strict = ec.top_similar(
        TopSimilarRequest(question="নতুন ভোটার", top_k=5, min_model_agreement=3, min_cosine=0.0)
    )
    assert strict.confident is False
    assert strict.tag == "unable_to_answer"


def test_single_model_request_does_not_force_a_fallback(ec):
    """The bar cannot exceed the models asked for, or one model always fails."""
    res = ec.top_similar(
        TopSimilarRequest(question="নতুন ভোটার হতে কি কি লাগে", top_k=5, models=["e5"], min_cosine=0.0)
    )
    assert res.confident is True
    assert res.tag == "new_voter_docs"


def test_fallback_still_returns_the_evidence(ec, monkeypatch):
    """top_similar and tag_votes stay populated so a caller can see why."""
    monkeypatch.setattr(
        EcBotService,
        "_per_model_tags",
        lambda self, similar: {"e5": "a", "qwen3": "b", "bge_m3": "c"},
    )
    res = ec.top_similar(TopSimilarRequest(question="নতুন ভোটার", top_k=5, min_cosine=0.0))
    assert res.tag == "unable_to_answer"
    assert res.top_similar and res.tag_votes


def test_weak_match_falls_back_even_when_models_agree(ec):
    """The failure the agreement gate cannot see: all three models drift to the
    same wrong tag because ANN always returns *some* nearest neighbour."""
    res = ec.top_similar(
        TopSimilarRequest(question="নতুন ভোটার হতে কি কি লাগে", top_k=5, min_cosine=0.99)
    )
    assert res.confident is False
    assert res.fallback_reason == "low_similarity"
    assert res.tag == "unable_to_answer"
    assert res.answer == ANSWERS["unable_to_answer"]


def test_floor_of_zero_admits_everything(ec):
    res = ec.top_similar(
        TopSimilarRequest(question="নতুন ভোটার হতে কি কি লাগে", top_k=5, min_cosine=0.0)
    )
    assert res.confident is True
    assert res.fallback_reason is None


def test_top_cosine_is_measured_on_the_winning_tag(ec):
    res = ec.top_similar(
        TopSimilarRequest(question="নতুন ভোটার হতে কি কি লাগে", top_k=5, min_cosine=0.0)
    )
    winning = [h.cosine_similarity for h in res.top_similar if h.tag == res.tag]
    assert res.top_cosine == pytest.approx(max(winning))


def test_low_similarity_outranks_disagreement_as_the_reason(ec, monkeypatch):
    """A question that matched nothing is not a disagreement problem."""
    monkeypatch.setattr(
        EcBotService,
        "_per_model_tags",
        lambda self, similar: {"e5": "a", "qwen3": "b", "bge_m3": "c"},
    )
    res = ec.top_similar(
        TopSimilarRequest(question="নতুন ভোটার", top_k=5, min_cosine=0.99)
    )
    assert res.fallback_reason == "low_similarity"
