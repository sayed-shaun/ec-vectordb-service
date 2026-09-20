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


from src.api.v1.qa.schema import QaTurn


def test_history_is_echoed_with_this_turn_appended(qa):
    res = qa.answer(QaRequest(question="নতুন ভোটার হতে কি কি লাগে?", min_cosine=0.0))
    assert [t.question for t in res.history] == ["নতুন ভোটার হতে কি কি লাগে?"]
    assert res.history[-1].tag == res.tag

    nxt = qa.answer(
        QaRequest(question="সংশোধন ফি কত টাকা", history=res.history, min_cosine=0.0)
    )
    assert [t.question for t in nxt.history] == [
        "নতুন ভোটার হতে কি কি লাগে?",
        "সংশোধন ফি কত টাকা",
    ]


def test_long_self_contained_question_ignores_history(qa):
    """Only bare questions borrow context; a full sentence stands on its own."""
    long_q = "নতুন ভোটার হতে হলে কি কি কাগজপত্র জমা দিতে হয় বলুন তো"
    assert len(long_q.split()) > 6

    bare = qa.answer(QaRequest(question=long_q, min_cosine=0.0))
    with_hist = qa.answer(
        QaRequest(
            question=long_q,
            history=[QaTurn(question="সংশোধন ফি কত টাকা", tag="fee_amount")],
            min_cosine=0.0,
        )
    )
    assert with_hist.used_context is False
    assert bare.tag == with_hist.tag


def test_short_question_uses_context_even_when_it_answered_confidently(qa, monkeypatch):
    """The failures this exists for are confidently wrong, not abstentions.

    A bare "ফি কত?" matches canonical questions literally titled "ফি কত" and
    returns the wrong tag at high cosine, so gating the retry on confidence
    would never fire.
    """
    seen = []
    real = EcBotService.top_similar

    def fake(self, req):
        seen.append(req.question)
        return real(self, req)

    monkeypatch.setattr(EcBotService, "top_similar", fake)
    res = qa.answer(
        QaRequest(
            question="ফি কত",
            history=[QaTurn(question="সংশোধন ফি কত টাকা", tag="fee_amount")],
            min_cosine=0.0,
        )
    )
    assert seen == ["ফি কত", "সংশোধন ফি কত টাকা ফি কত"], "expanded query must be tried"
    assert res.used_context is True


def test_context_rescues_a_bare_followup(qa, monkeypatch):
    """'How much?' carries no subject; the prior turn supplies it.

    The control flow is asserted directly rather than through the hash
    embedders in conftest, whose cosines do not behave like the real models'.
    """
    seen = []
    real = EcBotService.top_similar

    def fake(self, req):
        seen.append(req.question)
        res = real(self, req)
        # Confident only once the prior question has been folded in.
        object.__setattr__(res, "confident", req.question != "কত টাকা")
        return res

    monkeypatch.setattr(EcBotService, "top_similar", fake)
    res = qa.answer(
        QaRequest(
            question="কত টাকা",
            history=[QaTurn(question="সংশোধন ফি কত টাকা", tag="fee_amount")],
        )
    )
    assert seen == ["কত টাকা", "সংশোধন ফি কত টাকা কত টাকা"]
    assert res.used_context is True
    assert res.confident is True


def test_context_window_is_bounded(qa):
    """Only the last ECVDB_CONTEXT_TURNS questions are folded in."""
    from src.core.config import settings

    hist = [QaTurn(question=f"প্রশ্ন{i}", tag="t") for i in range(6)]
    built = EcBotService._with_context("কত টাকা", hist)
    kept = [f"প্রশ্ন{i}" for i in range(6 - settings.context_turns, 6)]
    assert built == " ".join(kept + ["কত টাকা"])
    assert "প্রশ্ন0" not in built


def test_context_builder_is_a_noop_without_history(qa):
    assert EcBotService._with_context("কত টাকা", []) == "কত টাকা"


def test_empty_history_behaves_exactly_like_no_history(qa):
    a = qa.answer(QaRequest(question="একাউন্ট লক", history=[], min_cosine=0.0))
    b = qa.answer(QaRequest(question="একাউন্ট লক", min_cosine=0.0))
    assert (a.tag, a.used_context) == (b.tag, b.used_context)
