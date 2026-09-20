from src.fusion import Candidate, reciprocal_rank_fusion, weighted_score_fusion


def lists(**kwargs) -> dict[str, list[Candidate]]:
    return {
        model: [Candidate(doc_id=d, score=1.0 - i * 0.1, rank=i + 1) for i, d in enumerate(docs)]
        for model, docs in kwargs.items()
    }


def test_unanimous_document_beats_single_model_favourite():
    per_model = lists(e5=["a", "b"], qwen3=["b", "c"], bge_m3=["b", "c"])
    fused = reciprocal_rank_fusion(per_model, k=60, top_k=3)
    assert fused[0].doc_id == "b"
    assert fused[0].votes == 3


def test_votes_and_ranks_are_reported_per_model():
    per_model = lists(e5=["x", "y"], qwen3=["y"], bge_m3=["y"])
    fused = reciprocal_rank_fusion(per_model, top_k=5)
    top = fused[0]
    assert top.doc_id == "y"
    assert top.per_model_rank == {"e5": 2, "qwen3": 1, "bge_m3": 1}
    assert set(top.per_model_score) == {"e5", "qwen3", "bge_m3"}


def test_min_votes_drops_documents_only_one_model_found():
    per_model = lists(e5=["only_e5", "shared"], qwen3=["shared"], bge_m3=["shared"])
    fused = reciprocal_rank_fusion(per_model, top_k=10, min_votes=2)
    assert [h.doc_id for h in fused] == ["shared"]


def test_weights_shift_the_ranking():
    per_model = lists(e5=["a"], qwen3=["b"], bge_m3=["b"])
    unweighted = reciprocal_rank_fusion(per_model, top_k=2)
    assert unweighted[0].doc_id == "b"

    weighted = reciprocal_rank_fusion(per_model, weights={"e5": 10.0}, top_k=2)
    assert weighted[0].doc_id == "a"


def test_weighted_score_fusion_normalises_incomparable_scales():
    per_model = {
        "e5": [Candidate("a", 0.91, 1), Candidate("b", 0.90, 2), Candidate("c", 0.89, 3)],
        "qwen3": [Candidate("b", 0.50, 1), Candidate("c", 0.30, 2), Candidate("a", 0.10, 3)],
    }
    fused = weighted_score_fusion(per_model, top_k=3)
    assert [h.doc_id for h in fused] == ["b", "a", "c"]


def test_empty_input_is_not_an_error():
    assert reciprocal_rank_fusion({}, top_k=5) == []
    assert weighted_score_fusion({"e5": []}, top_k=5) == []
