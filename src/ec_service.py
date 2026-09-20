"""EC FAQ retrieval: question matching, tag voting, answer resolution."""

from __future__ import annotations

import logging
import time
from collections import defaultdict

from src.api.v1.ec_bot.schema import (
    SimilarQuestion,
    TagVote,
    TopSimilarRequest,
    TopSimilarResponse,
)
from src.api.v1.qa.schema import QaRequest, QaResponse, QaTurn
from src.schemas import QueryRequest
from src.core.config import settings
from src.ec import FALLBACK_TAG, AnswerStore
from src.normalize import normalize
from src.service import RetrievalService

log = logging.getLogger(__name__)


class EcBotService:
    def __init__(self, retrieval: RetrievalService, answers: AnswerStore) -> None:
        self.retrieval = retrieval
        self.answers = answers

    def top_similar(self, req: TopSimilarRequest) -> TopSimilarResponse:
        started = time.perf_counter()
        normalized = normalize(req.question, req.normalizer)

        inner = self.retrieval.query(
            QueryRequest(
                question=normalized,
                top_k=req.top_k,
                models=req.models,
                fusion=req.fusion,
                min_votes=req.min_votes,
                include_text=True,
            )
        )

        similar: list[SimilarQuestion] = []
        for rank, hit in enumerate(inner.hits, 1):
            cosines = [v.score for v in hit.per_model.values()]
            similar.append(
                SimilarQuestion(
                    rank=rank,
                    question=hit.text or "",
                    tag=hit.metadata.get("tag", ""),
                    cosine_similarity=round(sum(cosines) / len(cosines), 6) if cosines else 0.0,
                    votes=hit.votes,
                    per_model=dict(hit.per_model),
                )
            )

        tag_votes = self._vote_tags(similar, inner.hits)
        winner = tag_votes[0] if tag_votes else None

        per_model_tag = self._per_model_tags(similar)
        agreement = sum(1 for t in per_model_tag.values() if winner and t == winner.tag)

        required = min(
            req.min_model_agreement or settings.min_model_agreement,
            len(inner.models) or 1,
        )
        floor = settings.min_cosine if req.min_cosine is None else req.min_cosine
        top_cosine = (
            max((sq.cosine_similarity for sq in similar if winner and sq.tag == winner.tag),
                default=0.0)
        )

        if winner is None:
            fallback_reason = "no_match"
        elif top_cosine < floor:
            fallback_reason = "low_similarity"
        elif agreement < required:
            fallback_reason = "model_disagreement"
        else:
            fallback_reason = None

        confident = fallback_reason is None

        if confident:
            tag = winner.tag
            answer = self.answers.get(tag) if req.include_answer else None
        else:
            tag = FALLBACK_TAG
            answer = self.answers.fallback() if req.include_answer else None
            log.info(
                "%s for %r: top_cosine=%.3f floor=%.2f, %d/%d models agreed, per-model %s",
                fallback_reason, normalized, top_cosine, floor,
                agreement, required, per_model_tag,
            )

        return TopSimilarResponse(
            input_question=req.question,
            normalized_question=normalized,
            top_similar=similar,
            tag=tag,
            answer=answer,
            tag_votes=tag_votes,
            took_ms=round((time.perf_counter() - started) * 1000, 2),
            per_model_tag=per_model_tag,
            model_agreement=agreement,
            confident=confident,
            top_cosine=round(top_cosine, 6),
            fallback_reason=fallback_reason,
        )

    def answer(self, req: QaRequest) -> QaResponse:
        """Reduce a full retrieval to the one thing a front end needs.

        A follow-up like "how much is the fee?" names no subject, so it is
        retried against the prior turns when it fails on its own. The retry
        only ever runs after a failure, so carrying history cannot change an
        answer that already stood up without it.
        """
        def ask(question: str):
            return self.top_similar(
                TopSimilarRequest(
                    question=question,
                    top_k=req.top_k,
                    normalizer=req.normalizer,
                    min_model_agreement=req.min_model_agreement,
                    min_cosine=req.min_cosine,
                    include_answer=True,
                )
            )

        full = ask(req.question)
        used_context = False

        # A confident answer is not evidence the question was understood: a
        # bare "ফি কত?" matches canonical questions literally titled "ফি কত"
        # at 0.89 and lands on the wrong tag. Shape decides, not confidence.
        if req.history and (self._needs_context(req.question) or not full.confident):
            expanded = self._with_context(req.question, req.history)
            if expanded != req.question:
                retry = ask(expanded)
                if retry.confident:
                    log.info(
                        "context applied to %r -> %r: %s -> %s",
                        req.question, expanded, full.tag, retry.tag,
                    )
                    full, used_context = retry, True

        history = list(req.history) + [QaTurn(question=req.question, tag=full.tag)]

        return QaResponse(
            question=req.question,
            answer=full.answer,
            tag=full.tag,
            confident=full.confident,
            model_agreement=full.model_agreement,
            per_model_tag=full.per_model_tag,
            top_cosine=full.top_cosine,
            fallback_reason=full.fallback_reason,
            matched_question=full.top_similar[0].question if full.top_similar else None,
            took_ms=full.took_ms,
            history=history,
            used_context=used_context,
        )

    @staticmethod
    def _needs_context(question: str) -> bool:
        """Whether a question is too bare to carry its own subject.

        Length is the signal that survives: the follow-ups in this corpus are
        overwhelmingly short elisions -- "ফি কত?", "কতদিন লাগে?", "কাগজপত্র কী
        কী?" -- rather than explicit anaphora, so a leading-pronoun test alone
        misses most of them.
        """
        return len(question.split()) <= settings.context_trigger_words

    @staticmethod
    def _with_context(question: str, history: list[QaTurn]) -> str:
        """Fold the last few questions in front of a follow-up.

        Only the caller's own questions are used. Tags are snake_case English
        identifiers and answers are long and bureaucratic -- neither sits in
        the same space as a short colloquial Bangla query, so both would pull
        the vector off the manifold rather than disambiguate it.
        """
        prior = [t.question.strip() for t in history[-settings.context_turns:] if t.question.strip()]
        return " ".join(prior + [question]) if prior else question

    def _per_model_tags(self, similar: list[SimilarQuestion]) -> dict[str, str]:
        """The tag each model would pick on the strength of its own ranking.

        A model votes across its whole returned list rather than on its rank-1
        hit alone: with ~34 questions sharing a tag, one fluke at the top
        should not decide whether the query is answerable at all.
        """
        per_model: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for sq in similar:
            if not sq.tag:
                continue
            for name, vote in sq.per_model.items():
                per_model[name][sq.tag] += 1.0 / (settings.rrf_k + vote.rank)

        return {
            name: max(tags.items(), key=lambda kv: kv[1])[0]
            for name, tags in per_model.items()
            if tags
        }

    def _vote_tags(self, similar: list[SimilarQuestion], hits) -> list[TagVote]:
        """Aggregate retrieved questions onto their tags.

        A tag owning four of the top ten is a far stronger signal than one
        owning only the single best hit, which is what top-1 discards.
        """
        score: dict[str, float] = defaultdict(float)
        hit_count: dict[str, int] = defaultdict(int)
        best_rank: dict[str, int] = {}

        for sq, hit in zip(similar, hits):
            if not sq.tag:
                continue
            score[sq.tag] += hit.score
            hit_count[sq.tag] += 1
            best_rank.setdefault(sq.tag, sq.rank)

        votes = [
            TagVote(
                tag=tag,
                score=round(value, 6),
                hits=hit_count[tag],
                best_rank=best_rank[tag],
                answer=self.answers.get(tag),
            )
            for tag, value in score.items()
        ]
        votes.sort(key=lambda v: (-v.score, v.best_rank))
        return votes
