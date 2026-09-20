# CLAUDE.md

## Rule zero: no hardcoded solutions

**Hardcoded solutions will not be accepted.** Solve the general case; anything
that only makes one input, test or demo pass is rejected, even if it is green.

- No special-casing a question, tag or string in `src/` — no `if question ==`,
  keyword tables or per-query overrides.
- No bypassing the pipeline: normalise → embed → ANN → RRF → tag vote → answer,
  through the similarity floor and consensus gate.
- No code that detects it is under test and branches differently.
- No inline magic numbers. Thresholds, `top_k`, RRF `k`, batch sizes, model ids,
  prefixes, dims, weights, paths → `src/core/config.py`, overridable by an env var of the same name.
- No corpus data in source; it is loaded from `data/`.
- No fixing data bugs (the `"tag"` placeholder, stub answers) in retrieval code —
  report them via `GET /ec_bot/audit`.
- No faked results: return the honest `unable_to_answer` fallback with a real
  `fallback_reason`.

If a general solution isn't reachable, say so instead of shipping a narrow one.
Tuning a config default is fine when it's backed by corpus-wide measurement
(see the floor tables in the README).

## Conventions

- Python ≥3.11, `from __future__ import annotations`, Pydantic for every
  request and response.
- Comments explain *why*; match the density already in the file.
- Config via `pydantic-settings`, one env var per setting, no prefix — no
  scattered `os.getenv`. Log with `logging.getLogger(__name__)`, never `print`.
- Must boot without a GPU: `resolved_device()` degrades to CPU, keep it so.
- Pinned versions in `pyproject.toml`; `torch>=2.6` is a hard floor.

## Commits

One concise line, imperative, capitalised, no full stop, ~72 chars, no body.
**No AI attribution** — no `Co-Authored-By: Claude`, no "Generated with Claude
Code", no 🤖 line. Say what the change does, not which files moved. Commit only
when asked.

```
Answer bare follow-ups by folding in prior questions
```

## Tests

`pytest`, or `pytest tests/test_qa.py -q`. Assert behaviour — `tag`,
`confident`, `fallback_reason`, ordering, gate behaviour — not memorised answer
strings, which is the hardcoding rule in test form. Test the general property
you fixed, not the input that reported it.

## Running

```
docker compose up -d                                                   # CPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d   # GPU
```

Local: `uvicorn main:app --reload` with Qdrant at `QDRANT_URL`. Run
`python scripts/ingest_ec.py` before querying an empty collection.
