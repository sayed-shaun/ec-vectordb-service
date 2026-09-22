"""The QA endpoint: one question in, one answer out."""

import pytest

from src.api.v1.qa.schema import QaRequest, QaTurn
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


def test_history_is_echoed_with_this_turn_appended(qa):
    res = qa.answer(QaRequest(question="নতুন ভোটার হতে কি কি লাগে?", min_cosine=0.0))
    assert [t.question for t in res.history] == ["নতুন ভোটার হতে কি কি লাগে?"]
    assert res.history[-1].answer == res.answer

    nxt = qa.answer(
        QaRequest(question="সংশোধন ফি কত টাকা", history=res.history, min_cosine=0.0)
    )
    assert [t.question for t in nxt.history] == [
        "নতুন ভোটার হতে কি কি লাগে?",
        "সংশোধন ফি কত টাকা",
    ]


def test_empty_history_behaves_exactly_like_no_history(qa):
    a = qa.answer(QaRequest(question="একাউন্ট লক", history=[], min_cosine=0.0))
    b = qa.answer(QaRequest(question="একাউন্ট লক", min_cosine=0.0))
    assert (a.tag, a.used_context) == (b.tag, b.used_context)


def test_placeholder_answer_is_refused(qa):
    """`fraction` ships the literal string "tag" as its answer, for 772
    questions. Routing to it is correct; serving it is not."""
    ec = EcBotService(qa.retrieval, AnswerStore({**ANSWERS, "stub_tag": "tag"}))
    assert ec._is_placeholder("stub_tag") is True
    assert ec._is_placeholder("account_locked") is False, "a real answer must survive"
    assert ec._is_placeholder("no_such_tag") is True, "a tag with no entry at all is a placeholder"


def test_fallback_text_reused_as_a_real_answer_is_a_placeholder(qa):
    """A tag whose answer is a verbatim copy of the disclaimer reads as a
    confident answer while saying nothing -- 16 tags in the real dataset,
    264 questions, distinct from the too-short-to-be-an-answer case."""
    disguised = AnswerStore({**ANSWERS, "smart_card_from_where": ANSWERS["unable_to_answer"]})
    ec = EcBotService(qa.retrieval, disguised)
    assert ec._is_placeholder("smart_card_from_where") is True
    assert ec._is_placeholder("unable_to_answer") is False, "the fallback tag is not a placeholder of itself"


# --- rewrite integration -------------------------------------------------
# The anchor heuristic (concatenating the last self-sufficient question) is
# gone; an external service does this instead. ec_service.answer() only has
# to call it correctly and degrade if it is unset or fails -- the quality of
# the rewrite itself is that service's concern, covered by src/test_rewrite.py.


def test_rewrite_is_a_noop_with_no_url_configured(qa):
    """Default settings ship an empty rewrite_url, so this must never depend
    on the external service being reachable -- not even the import graph
    touches a socket. No monkeypatching: this exercises the real function."""
    res = qa.answer(
        QaRequest(
            question="নতুন ভোটার হতে কি কি লাগে?",
            history=[QaTurn(question="আগের প্রশ্ন", answer="আগের উত্তর")],
            min_cosine=0.0,
        )
    )
    assert res.used_context is False
    assert res.confident is True


def test_rewritten_question_is_what_gets_retrieved(qa, monkeypatch):
    """The rewriter's output, not the caller's original wording, is searched
    -- but only once the bare question has already failed on its own.

    top_similar is fully stubbed rather than spied-on: the bare question must
    fail and the rewritten one must succeed, deterministically, which the
    real embeddings and corpus-size-dependent tag voting cannot guarantee.
    """
    from src.api.v1.ec_bot.schema import TopSimilarResponse

    seen = []

    def fake(self, req):
        seen.append(req.question)
        confident = req.question == "সংশোধন ফি কত টাকা"
        return TopSimilarResponse(
            input_question=req.question,
            normalized_question=req.question,
            top_similar=[],
            tag="fee_amount" if confident else "unable_to_answer",
            answer="২৩০ টাকা" if confident else None,
            confident=confident,
            fallback_reason=None if confident else "low_similarity",
        )

    monkeypatch.setattr(EcBotService, "top_similar", fake)
    monkeypatch.setattr(
        "src.ec_service.rewrite_question",
        lambda question, history, **k: ("সংশোধন ফি কত টাকা", True),
    )
    res = qa.answer(
        QaRequest(
            question="আর ফি?",
            history=[QaTurn(question="আমার nid ভুল আছে", answer="...")],
            min_cosine=0.0,
        )
    )
    assert seen == ["আর ফি?", "সংশোধন ফি কত টাকা"], "bare must be tried before the rewrite"
    assert res.used_context is True
    assert res.tag == "fee_amount"
    # The history entry keeps what the caller actually asked, not the rewrite.
    assert res.history[-1].question == "আর ফি?"


def test_rewrite_failure_falls_back_to_the_raw_question(qa, monkeypatch):
    """A rewrite failure must never become a request failure.

    The bare attempt is stubbed to fail -- with a real corpus and a floor of
    0.0 it would just succeed on its own, which would exercise nothing.
    """
    from src.api.v1.ec_bot.schema import TopSimilarResponse

    seen = []

    def fake(self, req):
        seen.append(req.question)
        return TopSimilarResponse(
            input_question=req.question, normalized_question=req.question,
            top_similar=[], tag="unable_to_answer", confident=False,
            fallback_reason="low_similarity",
        )

    monkeypatch.setattr(EcBotService, "top_similar", fake)
    monkeypatch.setattr(
        "src.ec_service.rewrite_question",
        lambda question, history, **k: (question, False),
    )
    res = qa.answer(
        QaRequest(
            question="নতুন ভোটার হতে কি কি লাগে?",
            history=[QaTurn(question="আগের প্রশ্ন", answer="...")],
            min_cosine=0.0,
        )
    )
    assert seen == ["নতুন ভোটার হতে কি কি লাগে?"], "rewrite declined, so no second attempt"
    assert res.used_context is False
    assert res.confident is False
    assert res.tag == "unable_to_answer"


def test_history_sent_to_the_rewriter_matches_its_schema(qa, monkeypatch):
    """{"question": ..., "answer": ...} is the exact shape the external
    service documents for a history entry. The bare attempt must fail first,
    or the rewriter is never consulted at all under the new bare-first order."""
    from src.api.v1.ec_bot.schema import TopSimilarResponse

    monkeypatch.setattr(
        EcBotService,
        "top_similar",
        lambda self, req: TopSimilarResponse(
            input_question=req.question, normalized_question=req.question,
            top_similar=[], tag="unable_to_answer", confident=False,
            fallback_reason="low_similarity",
        ),
    )
    captured = {}

    def fake(question, history, **k):
        captured["question"] = question
        captured["history"] = history
        return question, False

    monkeypatch.setattr("src.ec_service.rewrite_question", fake)
    qa.answer(
        QaRequest(
            question="দ্বিতীয় প্রশ্ন",
            history=[QaTurn(question="প্রথম প্রশ্ন", answer="প্রথম উত্তর")],
            min_cosine=0.0,
        )
    )
    assert captured["question"] == "দ্বিতীয় প্রশ্ন"
    assert captured["history"] == [{"question": "প্রথম প্রশ্ন", "answer": "প্রথম উত্তর"}]


# --- chat_id: server-held history, as an alternative to client-echoed history ---


@pytest.fixture(autouse=True)
def clean_sessions():
    from src.core.memory import store
    store._sessions.clear()
    yield
    store._sessions.clear()


def test_chat_id_is_echoed_back(qa):
    res = qa.answer(QaRequest(question="নতুন ভোটার হতে কি কি লাগে?", chat_id="c1", min_cosine=0.0))
    assert res.chat_id == "c1"


def test_chat_id_accumulates_across_calls_without_the_caller_holding_state(qa):
    """The point of chat_id: the caller sends only `question` and the id."""
    qa.answer(QaRequest(question="প্রথম প্রশ্ন", chat_id="c1", min_cosine=0.0))
    res = qa.answer(QaRequest(question="দ্বিতীয় প্রশ্ন", chat_id="c1", min_cosine=0.0))
    assert [t.question for t in res.history] == ["প্রথম প্রশ্ন", "দ্বিতীয় প্রশ্ন"]


def test_different_chat_ids_do_not_see_each_others_history(qa):
    qa.answer(QaRequest(question="চ্যাট এ প্রশ্ন", chat_id="a", min_cosine=0.0))
    res = qa.answer(QaRequest(question="চ্যাট বি প্রশ্ন", chat_id="b", min_cosine=0.0))
    assert [t.question for t in res.history] == ["চ্যাট বি প্রশ্ন"]


def test_chat_id_takes_priority_over_client_supplied_history(qa, monkeypatch):
    """Only one source of truth per call -- an explicit choice, not an
    accident of whichever field happens to get read first. Seeded directly
    via the store rather than through qa.answer(), so the only thing under
    test is which source answer() reads from."""
    from src.api.v1.ec_bot.schema import TopSimilarResponse
    from src.core.memory import store

    store.append("c1", "সীড প্রশ্ন", "সীড উত্তর")

    monkeypatch.setattr(
        EcBotService,
        "top_similar",
        lambda self, req: TopSimilarResponse(
            input_question=req.question, normalized_question=req.question,
            top_similar=[], tag="unable_to_answer", confident=False,
            fallback_reason="low_similarity",
        ),
    )
    captured = {}

    def fake(q, h, **k):
        captured["history"] = h
        return q, False

    monkeypatch.setattr("src.ec_service.rewrite_question", fake)
    qa.answer(
        QaRequest(
            question="প্রশ্ন",
            chat_id="c1",
            history=[QaTurn(question="এটা উপেক্ষিত হবে", answer="...")],
            min_cosine=0.0,
        )
    )
    assert captured["history"] == [{"question": "সীড প্রশ্ন", "answer": "সীড উত্তর"}], (
        "chat_id's own history must win over client-supplied `history`"
    )


def test_no_chat_id_and_no_history_behaves_as_a_single_turn(qa):
    res = qa.answer(QaRequest(question="নতুন ভোটার হতে কি কি লাগে?", min_cosine=0.0))
    assert res.chat_id is None
    assert len(res.history) == 1


def test_bare_question_never_calls_the_rewriter_when_it_already_succeeds(qa, monkeypatch):
    """Trying bare first means a mid-conversation question that doesn't need
    rewriting skips the external call entirely -- both a latency win and
    what makes `used_context` a meaningful signal rather than 'history was
    present', regardless of whether it was actually needed."""
    called = []
    monkeypatch.setattr(
        "src.ec_service.rewrite_question",
        lambda *a, **k: called.append(1) or ("should never run", True),
    )
    res = qa.answer(
        QaRequest(
            question="নতুন ভোটার হতে কি কি লাগে?",
            history=[QaTurn(question="আগের প্রশ্ন", answer="আগের উত্তর")],
            min_cosine=0.0,
        )
    )
    assert not called, "the bare question already succeeded; rewriting it would be wasted work"
    assert res.used_context is False
    assert res.rewritten_question is None
    assert res.confident is True
