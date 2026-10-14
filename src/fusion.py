"""Fusion of the per-model candidate lists.

Ranks are fused rather than cosines because different models score on
incomparable scales. With a single model RRF reduces to its own ranking.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Candidate:
    """One hit from one model."""

    doc_id: str
    score: float
    rank: int
    payload: dict = field(default_factory=dict)


@dataclass
class FusedHit:
    doc_id: str
    fused_score: float
    votes: int
    per_model_rank: dict[str, int]
    per_model_score: dict[str, float]
    payload: dict = field(default_factory=dict)


def reciprocal_rank_fusion(
    per_model: dict[str, list[Candidate]],
    weights: dict[str, float] | None = None,
    k: int = 60,
    top_k: int = 10,
    min_votes: int = 1,
) -> list[FusedHit]:
    """Fuse per-model candidate lists into one ranking.

    ``k`` damps the head of each list, so cross-model agreement beats one
    model's top spot. ``min_votes`` drops hits too few models found.
    """
    weights = weights or {}
    acc: dict[str, FusedHit] = {}

    for model_name, candidates in per_model.items():
        weight = weights.get(model_name, 1.0)
        for cand in candidates:
            hit = acc.get(cand.doc_id)
            if hit is None:
                hit = FusedHit(
                    doc_id=cand.doc_id,
                    fused_score=0.0,
                    votes=0,
                    per_model_rank={},
                    per_model_score={},
                    payload=cand.payload,
                )
                acc[cand.doc_id] = hit
            elif not hit.payload and cand.payload:
                hit.payload = cand.payload

            hit.fused_score += weight / (k + cand.rank)
            hit.votes += 1
            hit.per_model_rank[model_name] = cand.rank
            hit.per_model_score[model_name] = cand.score

    hits = [h for h in acc.values() if h.votes >= min_votes]
    hits.sort(key=lambda h: (-h.fused_score, -h.votes, min(h.per_model_rank.values())))
    return hits[:top_k]


def weighted_score_fusion(
    per_model: dict[str, list[Candidate]],
    weights: dict[str, float] | None = None,
    k: int = 60,
    top_k: int = 10,
    min_votes: int = 1,
) -> list[FusedHit]:
    """Min-max normalise each model's scores, then take a weighted mean.

    The alternative to RRF for callers who need score magnitude.
    """
    weights = weights or {}
    acc: dict[str, FusedHit] = {}
    total_weight = sum(weights.get(name, 1.0) for name in per_model) or 1.0

    for model_name, candidates in per_model.items():
        if not candidates:
            continue
        weight = weights.get(model_name, 1.0)
        scores = [c.score for c in candidates]
        lo, hi = min(scores), max(scores)
        span = hi - lo

        for cand in candidates:
            norm = 1.0 if span == 0 else (cand.score - lo) / span
            hit = acc.setdefault(
                cand.doc_id,
                FusedHit(cand.doc_id, 0.0, 0, {}, {}, cand.payload),
            )
            if not hit.payload and cand.payload:
                hit.payload = cand.payload
            hit.fused_score += weight * norm / total_weight
            hit.votes += 1
            hit.per_model_rank[model_name] = cand.rank
            hit.per_model_score[model_name] = cand.score

    hits = [h for h in acc.values() if h.votes >= min_votes]
    hits.sort(key=lambda h: (-h.fused_score, -h.votes, min(h.per_model_rank.values())))
    return hits[:top_k]
