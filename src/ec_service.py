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
from src.core.memory import store as session_store
from src.rewrite import rewrite_question
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
        elif self._is_placeholder(winner.tag):
            # Routing was right; the dataset has no real answer to give. Two
            # shapes of this slipped into tag_answer.json: a stub too short to
            # be an answer ("tag", 3 chars, on 772 questions under `fraction`),
            # and a tag whose answer is a verbatim copy of the "call 105"
            # disclaimer (16 tags, 264 questions) -- which reads as a
            # confident answer while saying nothing at all.
            log.warning("tag %r has a placeholder answer; refusing it", winner.tag)
            fallback_reason = "placeholder_answer"
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

        The question is tried as asked first. Only a question that fails on
        its own -- fails the floor, the agreement gate, or lands on a
        placeholder answer -- goes to the external rewrite service, which
        turns it into a standalone question against the conversation so far.
        That conversation comes from ``chat_id`` (server-held, in memory)
        when given, else from the caller-echoed ``history``.

        Trying bare first is what lets a question that happens to arrive
        mid-conversation, but doesn't need it, skip the external call
        entirely -- and it is what lets a corpus tag ever win on a bare
        match at all. A fragment like "কী কাগজ লাগবে?" that names no subject
        of its own will fail and fall through to the rewriter; a question
        that already stands on its own never pays that cost.
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

        if req.chat_id:
            history_pairs = session_store.get(req.chat_id)
        else:
            history_pairs = [{"question": t.question, "answer": t.answer or ""} for t in req.history]

        full = ask(req.question)
        used_context = False
        rewritten_question: str | None = None

        if history_pairs and not full.confident:
            rewritten, was_rewritten = rewrite_question(
                req.question,
                history_pairs,
                url=settings.rewrite_url,
                timeout=settings.rewrite_timeout_s,
            )
            # used_context means "a rescue was attempted", independent of
            # whether it worked -- confident/fallback_reason already say
            # that. A rewrite that still lands on a placeholder tag (a real
            # case: "how many days to reissue?" resolves correctly to
            # nid_reissue_process_duration, which is itself one of the
            # verbatim-disclaimer tags) is a meaningfully different outcome
            # from never having tried, and collapsing the two would hide it.
            used_context = was_rewritten and rewritten != req.question
            if used_context:
                rewritten_question = rewritten
                retry = ask(rewritten)
                if retry.confident:
                    full = retry

        if req.chat_id:
            session_store.append(req.chat_id, req.question, full.answer)
            history = [QaTurn(**t) for t in session_store.get(req.chat_id)]
        else:
            history = list(req.history) + [QaTurn(question=req.question, answer=full.answer)]

        # The winning tag comes from aggregate evidence across several
        # candidates (_vote_tags), not necessarily the single best-ranked
        # one -- so top_similar[0] can belong to a different tag than the
        # one being answered. Show the best-ranked candidate that actually
        # backs the winning tag, or none rather than a mismatched example.
        matched = next((sq for sq in full.top_similar if sq.tag == full.tag), None)

        return QaResponse(
            question=req.question,
            answer=full.answer,
            tag=full.tag,
            confident=full.confident,
            model_agreement=full.model_agreement,
            per_model_tag=full.per_model_tag,
            top_cosine=full.top_cosine,
            fallback_reason=full.fallback_reason,
            matched_question=matched.question if matched else None,
            rewritten_question=rewritten_question,
            took_ms=full.took_ms,
            history=history,
            used_context=used_context,
            chat_id=req.chat_id,
        )

    def _is_placeholder(self, tag: str) -> bool:
        """Whether a tag's answer is a stub that escaped into published data.

        Two independent tells, since one misses the other's cases: too short
        to be a real answer, or an exact copy of the fallback disclaimer used
        as if it were a routed tag's own answer.
        """
        answer = self.answers.get(tag)
        if answer is None or len(answer.strip()) <= settings.min_answer_chars:
            return True
        fallback = self.answers.fallback()
        return bool(fallback) and answer.strip() == fallback.strip() and tag != FALLBACK_TAG

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
