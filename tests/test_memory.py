"""InMemoryStore: TTL expiry, the count cap, isolation.

Each test builds its own store rather than sharing the module-level
singleton, so nothing here needs to reset global state between tests.
"""

from __future__ import annotations

import time

import pytest

from src.core.config import settings
from src.core.memory import InMemoryStore


@pytest.fixture
def store():
    return InMemoryStore()


def test_unknown_chat_id_is_empty(store):
    assert store.get("never-seen") == []


def test_append_then_get_round_trips_in_order(store):
    store.append("c1", "প্রথম প্রশ্ন", "প্রথম উত্তর")
    store.append("c1", "দ্বিতীয় প্রশ্ন", "দ্বিতীয় উত্তর")
    assert store.get("c1") == [
        {"question": "প্রথম প্রশ্ন", "answer": "প্রথম উত্তর"},
        {"question": "দ্বিতীয় প্রশ্ন", "answer": "দ্বিতীয় উত্তর"},
    ]


def test_none_answer_is_stored_as_empty_string(store):
    """A refused turn has no answer; the rewrite service's schema wants a
    string, not null, for this field."""
    store.append("c1", "প্রশ্ন", None)
    assert store.get("c1") == [{"question": "প্রশ্ন", "answer": ""}]


def test_stores_are_independent_instances(store):
    other = InMemoryStore()
    store.append("c1", "q", "a")
    assert other.get("c1") == [], "a fresh instance must not see another's data"


def test_chats_are_isolated(store):
    store.append("a", "qa", "aa")
    store.append("b", "qb", "ab")
    assert store.get("a") == [{"question": "qa", "answer": "aa"}]
    assert store.get("b") == [{"question": "qb", "answer": "ab"}]


def test_clear_forgets_a_chat(store):
    store.append("c1", "q", "a")
    store.clear("c1")
    assert store.get("c1") == []


def test_clear_of_unknown_chat_id_does_not_raise(store):
    store.clear("never-seen")


def test_idle_session_expires(store, monkeypatch):
    monkeypatch.setattr(settings, "session_ttl_s", 0.05)
    store.append("c1", "q", "a")
    assert store.get("c1"), "not expired yet"
    time.sleep(0.1)
    assert store.get("c1") == [], "idle past the TTL must be forgotten"


def test_touching_a_session_resets_its_idle_clock(store, monkeypatch):
    monkeypatch.setattr(settings, "session_ttl_s", 0.15)
    store.append("c1", "q1", "a1")
    time.sleep(0.08)
    store.append("c1", "q2", "a2")  # touched again before expiry
    time.sleep(0.08)
    assert len(store.get("c1")) == 2, "still alive: under 0.15s since the last touch"


def test_count_cap_evicts_the_oldest_first(store, monkeypatch):
    monkeypatch.setattr(settings, "session_max_count", 2)
    store.append("oldest", "q", "a")
    store.append("middle", "q", "a")
    store.append("newest", "q", "a")
    assert store.get("oldest") == [], "evicted to make room"
    assert store.get("middle") != []
    assert store.get("newest") != []


def test_count_reports_live_sessions(store):
    assert store.count() == 0
    store.append("c1", "q", "a")
    store.append("c2", "q", "a")
    assert store.count() == 2


def test_module_singleton_is_a_ready_instance():
    """The module also exposes a process-wide singleton, `store`, matching
    the `registry = EmbedderRegistry()` pattern in src/embedding.py."""
    from src.core.memory import store as singleton

    assert isinstance(singleton, InMemoryStore)
