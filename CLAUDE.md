# CLAUDE.md

## Rule zero: no hardcoded solutions

**Hardcoded solutions will not be accepted.** Solve the general case; anything
that only makes one input, test or demo pass is rejected, even if it is green.

- No special-casing a question, tag or string in `src/` — no `if question ==`,
  keyword tables or per-query overrides.
- No bypassing the pipeline: normalise → embed → ANN → RRF → top-k hits.
- No code that detects it is under test and branches differently.
- No inline magic numbers. Thresholds, `top_k`, RRF `k`, batch sizes, model ids,
  prefixes, dims, weights, paths → `src/core/config.py`, overridable by an env var of the same name.
- No corpus data in source; it is loaded from `data/`.
- No fixing data bugs in retrieval code — fix them in the corpus under `data/`.
- No faked results: return what the index actually retrieved.

If a general solution isn't reachable, say so instead of shipping a narrow one.
Tuning a config default is fine when it's backed by corpus-wide measurement
(see the measurements in the README).

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
Return each hit's tag alongside its cosine
```

## Tests

`pytest`, or `pytest tests/test_ec.py -q`. Assert behaviour — `tag`,
ordering, `top_k` bounds — not memorised corpus strings, which is the hardcoding rule in test form. Test the general property
you fixed, not the input that reported it.

## Running

```
docker compose up -d   # needs the NVIDIA container runtime; see docker-compose.yml for CPU
```

Local: `uvicorn main:app --reload` with Qdrant at `QDRANT_URL`. Run
`python scripts/ingest_ec.py` before querying an empty collection.
