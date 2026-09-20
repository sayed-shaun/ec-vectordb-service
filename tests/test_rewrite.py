"""The rewrite client: a best-effort call to an external service.

A rewrite failure must never become a request failure -- every failure mode
here degrades to the raw question, not an exception.
"""

from __future__ import annotations

import json
import urllib.error

import pytest

from src.rewrite import rewrite_question


def test_disabled_when_no_url_is_configured():
    """No URL, no call -- not even an attempt, so tests and local dev never
    need the external service reachable."""
    result = rewrite_question(
        "কত টাকা", [{"question": "ফি", "answer": "..."}], url="", timeout=1.0
    )
    assert result == ("কত টাকা", False)


def test_skipped_when_history_is_empty():
    """Nothing to rewrite against on the first turn of a conversation."""
    result = rewrite_question("প্রথম প্রশ্ন", [], url="http://example.invalid/rewrite", timeout=1.0)
    assert result == ("প্রথম প্রশ্ন", False)


def test_successful_rewrite(monkeypatch):
    def fake_urlopen(request, timeout):
        assert timeout == 2.0
        body = json.loads(request.data)
        assert body == {
            "question": "কত টাকা",
            "history": [{"question": "ফি লাগবে", "answer": "হ্যাঁ"}],
        }

        class Resp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return json.dumps(
                    {"question": "কত টাকা", "rewritten_question": "ফি কত টাকা", "was_rewritten": True}
                ).encode()

        return Resp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    result = rewrite_question(
        "কত টাকা",
        [{"question": "ফি লাগবে", "answer": "হ্যাঁ"}],
        url="http://example.invalid/rewrite",
        timeout=2.0,
    )
    assert result == ("ফি কত টাকা", True)


def test_not_rewritten_response_is_passed_through(monkeypatch):
    """A self-contained follow-up: the service itself declines to rewrite."""

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(
                {"question": "q", "rewritten_question": "q", "was_rewritten": False}
            ).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Resp())
    result = rewrite_question(
        "q", [{"question": "p", "answer": "a"}], url="http://x/rewrite", timeout=1.0
    )
    assert result == ("q", False)


@pytest.mark.parametrize(
    "raiser",
    [
        lambda: (_ for _ in ()).throw(urllib.error.URLError("connection refused")),
        lambda: (_ for _ in ()).throw(TimeoutError("timed out")),
        lambda: (_ for _ in ()).throw(ValueError("bad json")),
        lambda: (_ for _ in ()).throw(OSError("network unreachable")),
    ],
)
def test_every_failure_mode_falls_back_to_the_raw_question(monkeypatch, raiser):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: raiser())
    result = rewrite_question(
        "আসল প্রশ্ন", [{"question": "আগেরটা", "answer": "..."}],
        url="http://example.invalid/rewrite", timeout=0.5,
    )
    assert result == ("আসল প্রশ্ন", False)


def test_malformed_response_falls_back(monkeypatch):
    """A 200 with no usable ``rewritten_question`` must not crash the caller."""

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"unexpected": "shape"}).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Resp())
    result = rewrite_question(
        "প্রশ্ন", [{"question": "আগেরটা", "answer": "..."}],
        url="http://example.invalid/rewrite", timeout=1.0,
    )
    assert result == ("প্রশ্ন", False)


def test_blank_rewritten_question_falls_back(monkeypatch):
    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(
                {"rewritten_question": "   ", "was_rewritten": True}
            ).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Resp())
    result = rewrite_question(
        "প্রশ্ন", [{"question": "আগেরটা", "answer": "..."}],
        url="http://example.invalid/rewrite", timeout=1.0,
    )
    assert result == ("প্রশ্ন", False)
