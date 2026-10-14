import pytest

from src.core.config import MODEL_SPECS
from src.schemas import Document, QueryRequest

MODEL_NAMES = [s.name for s in MODEL_SPECS]

BANGLA_DOCS = [
    Document(
        doc_id="d1",
        text="বাংলাদেশের রাজধানী ঢাকা একটি জনবহুল শহর",
        metadata={"source": "geo", "lang": "bn"},
    ),
    Document(
        doc_id="d2",
        text="পদ্মা সেতু বাংলাদেশের দীর্ঘতম সেতু",
        metadata={"source": "infra", "lang": "bn"},
    ),
    Document(
        doc_id="d3",
        text="ভোটার তালিকা হালনাগাদ কার্যক্রম শুরু হয়েছে",
        metadata={"source": "ec", "lang": "bn"},
    ),
]


@pytest.fixture
def seeded(service):
    service.ingest(BANGLA_DOCS)
    return service


def test_ingest_writes_one_point_per_document(service):
    count, models = service.ingest(BANGLA_DOCS)
    assert count == 3
    assert models == MODEL_NAMES
    assert service.store.count() == 3


def test_reingesting_same_doc_id_overwrites_rather_than_duplicates(service):
    service.ingest(BANGLA_DOCS)
    service.ingest([Document(doc_id="d1", text="নতুন লেখা", metadata={"source": "geo"})])
    assert service.store.count() == 3


def test_query_returns_topk_with_per_model_votes(seeded):
    res = seeded.query(QueryRequest(question="ভোটার তালিকা হালনাগাদ", top_k=2))
    assert len(res.hits) == 2
    assert res.hits[0].doc_id == "d3"
    assert res.hits[0].votes == len(MODEL_NAMES)
    assert set(res.hits[0].per_model) == set(MODEL_NAMES)
    assert res.hits[0].text.startswith("ভোটার")
    assert res.hits[0].metadata["source"] == "ec"
    assert res.took_ms >= 0


def test_scores_are_monotonically_decreasing(seeded):
    res = seeded.query(QueryRequest(question="বাংলাদেশের সেতু", top_k=3))
    scores = [h.score for h in res.hits]
    assert scores == sorted(scores, reverse=True)


def test_filters_restrict_the_candidate_pool(seeded):
    res = seeded.query(
        QueryRequest(question="বাংলাদেশ", top_k=5, filters={"source": "infra"})
    )
    assert [h.doc_id for h in res.hits] == ["d2"]


def test_filters_accept_a_list_of_values(seeded):
    res = seeded.query(
        QueryRequest(question="বাংলাদেশ", top_k=5, filters={"source": ["infra", "ec"]})
    )
    assert set(h.doc_id for h in res.hits) <= {"d2", "d3"}


def test_model_subset_uses_only_those_models(seeded):
    subset = MODEL_NAMES[:1]
    res = seeded.query(QueryRequest(question="ঢাকা", top_k=3, models=subset))
    assert res.models == subset
    assert all(set(h.per_model) <= set(subset) for h in res.hits)
    assert all(h.votes <= len(subset) for h in res.hits)


def test_unknown_model_is_rejected(seeded):
    with pytest.raises(ValueError, match="unknown model"):
        seeded.query(QueryRequest(question="ঢাকা", models=["gpt_embed"]))


def test_include_text_false_omits_bodies(seeded):
    res = seeded.query(QueryRequest(question="ঢাকা", top_k=2, include_text=False))
    assert all(h.text is None for h in res.hits)
    assert all(h.metadata for h in res.hits)


def test_weighted_score_fusion_is_selectable(seeded):
    res = seeded.query(QueryRequest(question="পদ্মা সেতু", top_k=3, fusion="weighted_score"))
    assert res.fusion == "weighted_score"
    assert res.hits[0].doc_id == "d2"


def test_delete_removes_the_document(seeded):
    seeded.store.delete(["d2"])
    assert seeded.store.count() == 2
    res = seeded.query(QueryRequest(question="পদ্মা সেতু", top_k=5))
    assert "d2" not in [h.doc_id for h in res.hits]


def test_existing_doc_ids_reports_only_indexed_rows(seeded):
    found = seeded.store.existing_doc_ids(["d1", "d3", "nope"])
    assert found == {"d1", "d3"}


def test_existing_doc_ids_handles_empty_and_large_inputs(service):
    assert service.store.existing_doc_ids([]) == set()
    service.ingest([Document(doc_id=f"b{i}", text=f"প্রশ্ন {i}") for i in range(300)])
    found = service.store.existing_doc_ids([f"b{i}" for i in range(300)])
    assert len(found) == 300
