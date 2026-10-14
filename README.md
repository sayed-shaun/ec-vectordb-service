# EC VectorDB Service

Top-k retrieval service for the EC FAQ corpus. A question is normalised,
embedded with `intfloat/multilingual-e5-large-instruct`, matched against
canonical questions in Qdrant, and the closest `top_k` come back with their
tag and cosine similarity.

The endpoint is `POST /topk/top_similar/`. The request and response fields match
the old EC bot service at `:8002`, but the path moved from `/ec_bot/top_similar/`,
so existing callers must update the URL.

## Corpus

`data/question_tag.csv` — 17,995 canonical questions over 532 tags. Only the
question text is embedded; the tag and the original surface form ride along in
the payload and come back with each hit. `data/` is fetched from a private
repo by `scripts/fetch_dataset.sh` and is not committed.

## Pipeline

```
question → normalise → embed (e5) → Qdrant ANN → RRF → top-k hits
```

The registry and fusion are keyed by model name, so adding a second embedding
model is a `MODEL_SPECS` entry in `src/core/config.py`, not a rewrite. With one
model, RRF preserves its ranking and `cosine_similarity` is that model's raw
cosine.

## API

### `POST /topk/top_similar/`

```json
{ "question": "আমার নাম ভুল আছে আমার nid কার্ডে কি করবো ?", "top_k": 10 }
```

| field | default | |
|---|---|---|
| `question` | required | |
| `top_k` | 10 | 1–100 |
| `models` | all configured | subset of model names |
| `fusion` | `rrf` | `rrf` or `weighted_score` |
| `min_votes` | 1 | drop hits fewer models retrieved |
| `normalizer` | `compat` | `compat`, `safe` or `none` |

Response:

```json
{
  "input_question": "...",
  "normalized_question": "...",
  "top_similar": [
    {"rank": 1, "question": "...", "tag": "...", "cosine_similarity": 0.93,
     "votes": 1, "per_model": {"e5": {"rank": 1, "score": 0.93}}}
  ],
  "took_ms": 41.2
}
```

### `POST /reindex`

Downloads `question_tag.csv` from the private GitHub repo (`DATASET_REPO`,
`DATASET_REF`, `DATASET_PATH`, using `GITHUB_TOKEN`), then embeds and upserts it
in the background. Returns `202` with the job status, or `409` if one is
already running.

- The download replaces `data/question_tag.csv` only if it parses to at least
  one row, so a failed fetch keeps the previous copy and the index untouched.
- Indexing upserts in place, so `top_similar` keeps answering throughout. Only
  after every row lands are points no longer in the corpus pruned.
- `GET /reindex` reports `status` (`idle`, `running`, `succeeded`, `failed`),
  `phase`, `indexed`/`total`, `pruned` and `error`.
- The job lives in this process's memory: one worker, and status is lost on
  restart.
- There is no authentication. Set `REINDEX_TOKEN` and send it as
  `X-Reindex-Token` (only `POST` needs it; `GET /reindex` stays open), or keep the service behind a gateway; left unset, anyone
  who can reach it can trigger a re-embed of the whole corpus.

### `GET /health`

Qdrant state plus per-model residency, device and dtype.

## Normalisation

`compat` (default) matches what the corpus was indexed with; `safe` is a
conservative cleanup; `none` embeds the text as given. Queries and the index
must use the same normaliser — `scripts/ingest_ec.py --normalizer` takes the
same choices.

## Layout

```
main.py                     ASGI entrypoint, lifespan wiring
src/
  api/v1/topk/            top_similar router + schemas
  api/v1/system/            health
  core/config.py            settings + MODEL_SPECS
  embedding.py              model registry, device/dtype resolution
  fusion.py                 RRF and weighted-score fusion
  store.py                  Qdrant wrapper
  service.py                embed → search → fuse
  topk_service.py             top_similar: normalise + shape hits
  ec.py                     corpus loader
  dataset.py, indexer.py, reindex.py   fetch from GitHub, embed + upsert, background job
  api/v1/reindex/           /reindex router + status schema
  normalize.py
scripts/                    ingest_ec.py, fetch_dataset.sh, demo.py
tests/
```

## Run it

```
docker compose up -d                                                   # CPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d   # GPU
docker compose exec api python scripts/ingest_ec.py                    # index
```

Local: `uvicorn main:app --reload` with Qdrant at `QDRANT_URL`, then
`python scripts/ingest_ec.py` (`--recreate` to rebuild, `--resume` to continue
after a crash). Ingest is idempotent per question id. Changing the model or
normaliser means re-ingesting.

Weights land on the `models` named volume, so a rebuild does not re-download
them. The model runs fp16 on GPU; a CPU-resident model stays fp32 whatever
`DTYPE` says.

## Configuration

Every setting has a default in `src/core/config.py` and is overridable by an
env var of the same name (no prefix, from the environment or `.env`).
`.env.example` lists only the ones that normally differ per environment:

| var | default | |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | |
| `DEVICE` | `cpu` | `cpu`, `cuda` or `auto` |

For `/reindex`: `GITHUB_TOKEN` (repo read) and optionally `REINDEX_TOKEN`.

Others worth knowing: `COLLECTION`, `DTYPE`, `DEVICE_OVERRIDES`,
`CANDIDATE_MULTIPLIER`, `RRF_K`, `EMBED_BATCH_SIZE`, `EAGER_LOAD_MODELS`,
`LOG_LEVEL`, `LOG_FILE`.

## Logging

Logs go to the console and, when `LOG_FILE` is set, to a size-capped file:
past `LOG_MAX_BYTES` it is truncated (or rotated, if `LOG_BACKUP_COUNT` > 0),
so the service cannot fill a disk. Empty `LOG_FILE` logs to the console only.

## Tests

```
pytest
```

Runs against an in-memory Qdrant with deterministic fake embedders — no model
downloads and no running Qdrant.

## Known constraints

- **torch must be ≥ 2.6.** transformers refuses to `torch.load` pickle
  checkpoints on older torch (CVE-2025-32434); the floor stays even though e5
  ships safetensors.
- **No authentication**, except the optional `REINDEX_TOKEN` on `/reindex`. Keep the service behind your own gateway.
