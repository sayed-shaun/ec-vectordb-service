"""The QA endpoint: one question in, one answer out."""

import pytest

from src.api.v1.qa.schema import QaRequest
from src.schemas import Document
from src.ec import AnswerStore
from src.ec_service import EcBotService
from tests.test_ec import ANSWERS, CORPUS


@pytest.fixture
def qa(service):
    service.ingest(
        [Document(doc_id=q, text=t, metadata={"tag": tag}) for q, t, tag in CORPUS]
    )
    return EcBotService(service, AnswerStore(ANSWERS))


def test_answers_a_known_question(qa):
    res = qa.answer(QaRequest(question="নতুন ভোটার হতে কি কি লাগে?", min_cosine=0.0))
    assert res.confident is True
    assert res.tag == "new_voter_docs"
    assert res.answer == ANSWERS["new_voter_docs"]
    assert res.matched_question


def test_disagreement_returns_the_fallback_not_a_guess(qa, monkeypatch):
    monkeypatch.setattr(
        EcBotService,
        "_per_model_tags",
        lambda self, similar: {"e5": "a", "qwen3": "b", "bge_m3": "c"},
    )
    res = qa.answer(QaRequest(question="নতুন ভোটার হতে কি কি লাগে?", min_cosine=0.0))
    assert res.confident is False
    assert res.tag == "unable_to_answer"
    assert res.answer == ANSWERS["unable_to_answer"]


def test_threshold_is_overridable_per_request(qa, monkeypatch):
    monkeypatch.setattr(
        EcBotService,
        "_per_model_tags",
        lambda self, similar: {"e5": "new_voter_docs", "qwen3": "new_voter_docs", "bge_m3": "x"},
    )
    assert qa.answer(QaRequest(question="নতুন ভোটার", min_cosine=0.0)).confident is True
    strict = qa.answer(QaRequest(question="নতুন ভোটার", min_model_agreement=3))
    assert strict.confident is False


def test_empty_question_is_rejected(qa):
    with pytest.raises(Exception):
        qa.answer(QaRequest(question=""))


def test_off_topic_question_is_refused_by_the_floor(qa):
    res = qa.answer(QaRequest(question="নতুন ভোটার হতে কি কি লাগে?", min_cosine=0.99))
    assert res.confident is False
    assert res.fallback_reason == "low_similarity"
    assert res.tag == "unable_to_answer"
    assert res.answer == ANSWERS["unable_to_answer"]


def test_qa_reports_the_cosine_it_judged_on(qa):
    res = qa.answer(QaRequest(question="নতুন ভোটার হতে কি কি লাগে?", min_cosine=0.0))
    assert 0.0 < res.top_cosine <= 1.0
