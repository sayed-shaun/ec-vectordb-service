# EC VectorDB Service

Top-k retrieval service for the EC FAQ corpus. A question is normalised,
embedded with `intfloat/multilingual-e5-large-instruct`, matched against
canonical questions in Qdrant, and the closest `top_k` come back with their
tag and cosine similarity.

The endpoint is `POST /api/v1/topk`. It replaces `/ec_bot/top_similar/` from the
old EC bot service at `:8002`, so existing callers must update the URL.

## Corpus

`data/question_tag.csv` — 18,020 canonical questions over 532 tags. Only the
question text is embedded; the tag and the original surface form ride along in
the payload and come back with each hit, together with the tag's answer from
`data/tag_answer.json`, stored as metadata at index time. `data/` is fetched from a private
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

### `POST /api/v1/topk`

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

Each hit carries `answer`, its tag's answer from `data/tag_answer.json` (`null`
when the tag has none). The answer is stored in the point's payload next to the
tag when the question is indexed, so queries never read the answers file and an
answer change reaches queries only after a reindex. There is no winning tag,
similarity floor or fallback.

Response:

```json
{
  "input_question": "hi",
  "normalized_question": "hi",
  "top_similar": [
    {"rank": 1, "question": "hlw", "answer": "...", "tag": "greetings",
     "cosine_similarity": 0.985506}
  ]
}
```

### `POST /reindex`

Downloads `question_tag.csv` and `tag_answer.json` from the private GitHub repo (`DATASET_REPO`,
`DATASET_REF`, `DATASET_PATH`) with the **caller's** GitHub token, then embeds
and upserts it in the background. Returns `202` with the job status, `401` if
the `X-GitHub-Token` header is missing, or `409` if a job is already running.

```
curl -X POST http://localhost:8000/reindex -H "X-GitHub-Token: <token>"
```

The token needs read access to the dataset repo (fine-grained: Contents,
read-only). It is used for the download and dropped: nothing stores it, and it
is not in the status or the logs. A token GitHub rejects fails the job in the
fetch step with nothing indexed, so the token is the only credential the
endpoint needs.

- The download replaces `data/question_tag.csv` only if it parses to at least
  one row, so a failed fetch keeps the previous copy and the index untouched.
- Indexing upserts in place, so `top_similar` keeps answering throughout. Only
  after every row lands are points no longer in the corpus pruned.
- `GET /reindex` reports `status` (`idle`, `running`, `succeeded`, `failed`),
  `phase`, `indexed`/`total`, `pruned` and `error`.
- The job lives in this process's memory: one worker, and status is lost on
  restart.
- The token travels in a header over plain HTTP, so use TLS or a trusted
  network between the caller and the service.

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
  api/v1/topk/              top-k router + schemas
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
No `.env` is needed: with nothing set it talks to Qdrant at
`http://localhost:6333` and picks the GPU if one is visible, else the CPU.
`.env.example` lists the two you are most likely to override:

| var | default | |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | (`http://qdrant:6333` under Compose) |
| `DEVICE` | `auto` | `auto`, `cpu` or `cuda` |

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
- **No authentication** on the service itself; `/reindex` is gated only by a valid GitHub token. Keep it behind your own gateway.
