# EC VectorDB Service

Retrieval service for the EC FAQ bot. A user question is embedded by **three
different models**, each retrieves its own candidate list from Qdrant, the
lists are fused by Reciprocal Rank Fusion, and the retrieved canonical
questions **vote on a tag** — whose answer is returned.

Drop-in compatible with the existing service at `:8002`
(`POST /ec_bot/top_similar/`), with additive fields on top.

## Corpus

| | |
|---|---|
| canonical questions | 17,995 (`data/question_tag.csv`) |
| tags | 532 reachable, 1,380 answer entries (`data/tag_answer.json`) |
| questions per tag | min 5, mean 33.8, max 772 |
| answer coverage | complete — every reachable tag has an answer |

`GET /ec_bot/audit` reports this live. Two standing data issues it surfaces:

> ⚠️ **`tag_answer.json["fraction"]` is the literal string `"tag"`** — a
> placeholder that reached published data. `fraction` is the *largest* tag in
> the corpus at 772 questions (4.3%). Together with six other stub answers,
> **978 questions (5.4%) resolve to a degenerate response.** Retrieval cannot
> fix this; the data needs a real answer.
>
> ℹ️ 848 answer tags have no questions pointing at them and are unreachable by
> retrieval (presumably reached via the classifier path).

## Why questions only are embedded

Only the **question** text goes into the vectors. The tag and the original
surface form ride along in the payload; the answer is looked up from the
winning tag at query time.

This is *symmetric* retrieval — a user question matched against canonical
questions — and folding the answer into the vector breaks it three ways:

1. **It collapses each tag's internal structure.** ~34 questions share a tag
   (772 for `fraction`). Appending the same answer to all of them makes those
   vectors near-duplicates dominated by shared answer text, erasing the
   distinct phrasings that make retrieval work.
2. **Answers live in a different register** — declarative and bureaucratic,
   against colloquial interrogative queries. Mixing pulls every vector off the
   manifold the query sits on.
3. **It would propagate the `fraction` bug** into 772 vectors as the literal
   token `"tag"`.

Length argues the same way: answers reach 822 characters and E5 truncates at
512 tokens, so the question itself could be clipped off the end.

The real counter-case — a query matching an *answer's* content when no
canonical question covers it — is better served by a **separate tag-level
answer index** (532 vectors, negligible) fused in as a fourth source, rather
than by contaminating the question vectors. The architecture supports that;
it is not built yet, and should be A/B'd rather than assumed.

## The three voters

| slot | model | params | why it's here |
|---|---|---|---|
| `e5` | `intfloat/multilingual-e5-large` | 560M | Strong multilingual baseline; trained with asymmetric `query:`/`passage:` prefixes (applied automatically). |
| `qwen3` | `Qwen/Qwen3-Embedding-0.6B` | 596M | LLM-based embedder, last-token pooling, instruct-style query prompt. Fails differently from E5. |
| `bge_m3` | `BAAI/bge-m3` | 566M | Bangla-strong, 8k context, different corpus and objective — the real source of voting diversity. |

Edit `MODEL_SPECS` in `src/core/config.py` to swap a slot or change a weight.
Prefixes and pooling are per-model properties there, not scattered through the
code.

## Architecture

One Qdrant collection, one point per question, **three named vectors per
point**:

```
point q000123 ── vector "e5"     (1024) ─┐
            ├── vector "qwen3"  (1024) ─┼─ payload: {tag, raw_question, lang, ...}
            └── vector "bge_m3" (1024) ─┘
```

Named vectors rather than three collections: a question can never exist under
one model and be missing under another — the failure mode that silently
breaks vote-based retrieval. All three searches go out in one
`query_batch_points` round trip, so the third model costs almost no latency.

```
question ─→ normalise ─→ 3× embed ─→ 3× ANN (top_k × 4) ─→ RRF ─→ tag vote ─→ answer
```

### Two gates before an answer is returned

A question is answered only if it passes both. Either failure returns the
`unable_to_answer` tag with `confident: false` and a `fallback_reason`.

**1. The similarity floor.** The winning tag's best hit must reach
`MIN_COSINE`. This is the gate that matters most, because ANN retrieval
always returns *some* nearest neighbour however far away it is — ask for a
biriyani recipe and you get a birth-certificate answer at cosine 0.65.

Measured on the complete 17,995-point index. In-domain is sampled by querying
a corpus question **with its own row excluded from the hits**, so the match is
the nearest tag-sibling — a real paraphrase, not a memorised self-match:

| | n | min | p10 | median | max |
|---|---|---|---|---|---|
| in-domain, nearest paraphrase | 25 | 0.806 | 0.809 | 0.873 | 0.925 |
| off-domain | 10 | 0.624 | 0.652 | 0.745 | 0.834 |

| floor | rejects real | catches off-domain |
|---|---|---|
| 0.70 | 0/25 | 4/10 |
| **0.78** | **0/25** | **7/10** |
| 0.80 | 0/25 | 7/10 |
| 0.85 | 9/25 (36%) | 10/10 |

`0.78` catches the same seven as `0.80` with four times the margin below the
in-domain minimum, which matters at n=25. The three it lets through score
0.80, 0.83, 0.83 — and two of those are not errors: "write me a poem" lands on
`warning_to_prank_caller` and a train-ticket question on `unable_to_answer`.
The corpus covers them.

> The in-domain proxy is optimistic: tag-siblings in this corpus are close
> paraphrases, while real users diverge further. If false refusals show up in
> production, lower the floor before touching anything else.

Two apparent off-domain failures are not failures: "write me a poem" scores
0.804 onto `warning_to_prank_caller`, and a train-ticket question resolves to
`unable_to_answer` on its own. The corpus covers both.

**2. The consensus gate.**

Each model also picks a tag on the strength of its own ranking, and at least
`MIN_MODEL_AGREEMENT` of the three must land on the same one. This
catches *ambiguity between two plausible EC topics*, which the floor cannot
see — both candidates match well, the models just disagree about which.

It does **not** catch off-domain questions. All three models index the same
corpus, so an unrelated query makes all three drift to the same arbitrary
nearest tag: "today's football score" wins 3/3 unanimous agreement on
`faster_voter_registration`. They agree because they are wrong in the same
direction. That is the floor's job, not this one's.

A model votes across its whole returned list, not on its rank-1 hit alone.
With ~34 questions sharing a tag, one fluke at the top should not decide
whether the question is answerable at all.

`per_model_tag` and `top_cosine` report what each model picked and how well it
matched, so a fallback can always be explained; `fallback_reason` is
`low_similarity`, `model_disagreement` or `no_match`. The threshold never exceeds the number of models actually queried,
or a single-model request would always fall back.

### Two levels of voting

**Models vote on questions** via RRF. Scores across the three models are on
incomparable scales, so ranks are fused rather than cosines:

```
score(q) = Σ  weight_m / (k + rank_m(q))        k = 60
```

`k=60` damps the head of each list, so cross-model agreement beats one model's
top spot.

**Questions vote on tags.** Retrieved questions are grouped by tag and their
fused scores summed. With ~34 questions per tag, a tag owning four of the top
ten is a far stronger signal than one owning only the single best hit — which
is exactly what a top-1 lookup discards. `tag_votes` exposes the full ranking.

## API

### `POST /qa`

The endpoint a chat or IVR front end should call. Everything but the answer is
there for logging.

```json
{"question": "আমার নাম ভুল আছে আমার nid কার্ডে কি করবো ?"}
```

```json
{
  "question": "আমার নাম ভুল আছে আমার nid কার্ডে কি করবো ?",
  "answer": "চাহিত তথ্যের স্বপক্ষে দলিলাদি দাখিল পূর্বক …",
  "tag": "card_information_correction",
  "confident": true,
  "model_agreement": 3,
  "top_cosine": 0.88,
  "fallback_reason": null,
  "per_model_tag": {"e5": "card_information_correction",
                    "qwen3": "card_information_correction",
                    "bge_m3": "card_information_correction"},
  "matched_question": "আমার এনআইডি কার্ডে নিজের নাম ভুল আছে কিভাবে ঠিক করব",
  "took_ms": 453.07
}
```

Branch on `confident`, not on the answer text: when it is false the answer is
the fallback copy and the call should escalate to a human.

#### Follow-up questions

Pass the previous turns back and a bare follow-up becomes answerable:

```json
{"question": "ফি কত?",
 "history": [{"question": "হারানো এনআইডি কার্ড পুনরায় তুলতে চাই",
              "tag": "card_damaged_how_to_get_new_card"}]}
```

Without it, `ফি কত?` routes to `fraction` — there are canonical questions
literally titled "ফি কত", and `fraction` is the largest tag in the corpus with
the stub answer. With it, the query becomes the prior question plus this one
and routes to `card_lost_new_card_fee`.

**The server stores nothing.** `history` comes back on every response; hand it
straight to the next call. That keeps the service horizontally scalable — any
instance can serve any turn.

The trigger is the question's *shape*, not its confidence. A bare follow-up
retrieves something confidently wrong rather than abstaining, so gating the
retry on `confident` would never fire: in the gold run, 275 one-turn failures
were **all** confident wrong routes and none were abstentions. Questions of
`CONTEXT_TRIGGER_WORDS` (6) tokens or fewer borrow context; longer ones
only do so if they failed alone, and `used_context` reports which happened.

### `POST /ec_bot/top_similar/`

Request — `question` and `top_k` match the existing service; the rest is
optional:

```json
{"question": "নতুন ভোটার হতে কি কি লাগে?", "top_k": 5,
 "models": ["e5","qwen3","bge_m3"], "fusion": "rrf",
 "min_votes": 1, "normalizer": "compat", "include_answer": true}
```

Response — `input_question`, `normalized_question` and the four documented
keys of each `top_similar` entry are byte-compatible with `:8002`. Everything
else is additive:

```json
{
  "input_question": "নতুন ভোটার হতে কি কি লাগে?",
  "normalized_question": "নতুন ভোটার হতে কি কি লাগে",
  "top_similar": [
    {"rank": 1, "question": "নতুন ভোটার হওয়ার জন্য কী করতে হয়",
     "tag": "wish_to_create_nid_smart_card_new", "cosine_similarity": 0.98,
     "votes": 3, "per_model": {"e5": {"rank": 1, "score": 0.987}}}
  ],
  "tag": "wish_to_create_nid_smart_card_new",
  "answer": "…",
  "tag_votes": [{"tag": "…", "score": 0.031, "hits": 3, "best_rank": 1, "answer": "…"}],
  "took_ms": 84.2
}
```

`cosine_similarity` is the **mean cosine across the models that retrieved that
question** — still a cosine in [0,1], better calibrated than any single
model's, and safe to threshold on. Ordering is RRF's, which is the point of
voting, so it is not strictly monotonic in this field.

| endpoint | purpose |
|---|---|
| `POST /qa` | Question in, answer out. The plain contract. |
| `POST /ec_bot/top_similar/` | Compatible FAQ retrieval + voted tag + answer. |
| `GET /ec_bot/audit` | Dataset integrity report. |
| `GET /health` | Qdrant state, model residency, device and dtype. |

Four endpoints, and that is deliberate. The service carries **no
authentication**, so every route it exposes is reachable by anything that can
reach the port. `POST /documents` and `POST /documents/delete` were an
unauthenticated write and delete surface that no caller used -- ingest runs
in-process through `scripts/ingest_ec.py`, never over HTTP -- so they are
gone, along with `POST /query` (no caller; `top_similar` covers it) and
`POST /warmup` (meaningless while `EAGER_LOAD_MODELS` is `true`).
`tests/test_api.py` pins the route set so it cannot quietly grow back.

Writes happen through the ingest script, on the host, with access to the
Qdrant port -- not through the API.

## Normalisation

`src/normalize.py` reproduces the reference service's `normalized_question`
**bit-for-bit**. The strip set was extracted empirically by probing every ASCII
and common Bangla punctuation mark against `:8002`, because the upstream
OpenAPI schema does not describe the field.

It has real asymmetries — `“` is stripped but `”` survives, `–` is stripped but
`—` survives, and `$%&;<>\_{}` all survive. These look like omissions in a
hand-written list. They are reproduced anyway: **the normaliser must agree with
whatever built the index**, and "more correct" here would mean "silently
mismatched". Punctuation is *deleted*, not replaced with a space, so
`ভোটার,তালিকা` becomes one token `ভোটারতালিকা`.

`normalizer: "safe"` is the better choice for a freshly built index — it
applies NFC and preserves token boundaries. Use it for both ingest and query or
neither.

## Layout

```
main.py                     ASGI assembly only — `uvicorn main:app`
src/
  api/
    deps.py                 get_service / get_ec_service
    v1/
      router.py             aggregates the service routers
      system/               router.py + schema.py   /health
      ec_bot/               router.py + schema.py   /ec_bot/*
      qa/                   router.py + schema.py   /qa
  core/
    config.py               settings + MODEL_SPECS (the three voters)
    logger.py               console + size-capped file logging
  embedding.py              model registry, device/dtype resolution, preflight
  store.py                  Qdrant: named vectors, filters, batch query
  service.py                embed → search → fuse
  fusion.py                 RRF and weighted-score fusion
  ec.py                     dataset loading + integrity audit
  ec_service.py             the tag vote and answer lookup
  normalize.py              bit-for-bit reproduction of the :8002 normaliser
  schemas.py                internal retrieval DTOs (not an HTTP contract)
scripts/                    dataset fetch, ingest, demo
```

Three rules hold this together:

- **`v1` is a source layout, not a URL prefix.** The routes stay unversioned —
  `/ec_bot/top_similar/`, `/query`, `/health` — because this service has to
  stand in for the one at `:8002`, whose paths are fixed. A breaking change
  would mount a `v2` package alongside under an explicit prefix rather than
  move anything that already ships.
- **`src/core/` imports nothing from the rest of the package**, and everything
  else imports it. There is no cycle to reason about.
- **Routers never import `main`.** The lifespan builds the service objects and
  hands them to `src/api/deps.py`; the app imports routers and never the
  reverse.
- **`src/api/` is not imported by the domain modules.** Retrieval's types live
  in `src/schemas.py` because no route exposes them; only the EC bot's
  request and response models are an HTTP contract.

## Run it

Everything runs in Docker. **`DEVICE` is the only setting that differs
between this workstation and the GPU server** — `cpu` here, `cuda` there (or
`auto` to detect). An explicit `cuda` degrades to CPU rather than crashing if
the runtime is missing, so the same image boots in both places.

```bash
./scripts/fetch_dataset.sh            # needs gh auth, or GITHUB_TOKEN in .env
docker compose up -d --build
docker compose exec api python scripts/ingest_ec.py    # ~18k × 3 models
```

On a GPU host, add the overlay:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

Weights (~3.5 GB) land on the `models` named volume, so a rebuild does not
re-download them.

## VRAM

Weight footprint, which is what actually binds — encoder embedders have no KV
cache, so batch size moves activations but not the floor:

| model | fp32 | fp16 |
|---|---|---|
| multilingual-e5-large | 2136 MiB | 1068 MiB |
| Qwen3-Embedding-0.6B | 2273 MiB | 1136 MiB |
| bge-m3 | 2161 MiB | 1080 MiB |
| **all three** | **6570 MiB** | **3285 MiB** |

Add ~300 MiB of CUDA context and ~300 MiB of activations at batch 16 × 512
tokens, so all three in fp16 need **~3885 MiB**. That is fractionally under a
4096 MiB card *on paper*, but only on a card with nothing else on it — the
desktop session on this box measured 655 MiB resident, which puts it out of
reach. This is why fp16-everything is not the shipped default.

**Split across devices instead** — this fits in ~2.2 GiB:

```bash
DEVICE=cuda
DTYPE=float16
DEVICE_OVERRIDES='{"bge_m3":"cpu"}'
```

A CPU-resident model stays fp32 regardless of `DTYPE`: x86 has no native
fp16 compute path, so half precision there costs speed and saves RAM you are
not short of. `GET /health` reports the effective device and dtype per model.

## Logging

Console plus a file, both at `LOG_LEVEL`. The file is **hard-capped at
10 MiB**: past that it is truncated in place and starts again, with a marker
line explaining the gap, so the service cannot fill a disk unattended.

The stdlib does not do this on its own — `RotatingFileHandler` with
`backupCount=0` closes the file on rollover and *reopens it in append mode*, so
it grows without bound; the cap only bites when there is somewhere to rotate
to. `src/core/logger.py` supplies a `CappedFileHandler` that truncates instead,
and `tests/test_logging.py` pins the behaviour.

Truncating discards the older half of the history at exactly the moment the cap
is hit. When a post-mortem needs to span that boundary, set
`LOG_BACKUP_COUNT=2` first — that restores ordinary stdlib rotation, at
`(backups + 1) × 10 MiB` on disk.

uvicorn's own loggers are re-pointed at the root logger, so startup and access
lines land in the same file rather than in uvicorn's private handlers. In
compose the file is bind-mounted to `./logs/` so it outlives the container; set
`LOG_FILE=` (empty) to log to the console only, which is what you want
when a log driver already collects stdout.

## Tests

```bash
pip install -e ".[dev]"
pytest             # 92 tests, no model downloads, no running Qdrant needed
```

Tests swap in deterministic hash embedders and qdrant-client's in-memory mode,
which runs the real query and filter logic in-process — so everything but the
neural nets is genuinely exercised. `tests/test_normalize.py` pins all 13
normalisation cases captured from the live reference service; if one fails,
the index and queries have diverged.

## Configuration

One env var per setting, no prefix, from env or `.env`:

| var | default | |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | |
| `COLLECTION` | `ec_documents` | |
| `DEVICE` | `cpu` | `cuda` to use the GPU |
| `DTYPE` | `float32` | `float16` halves VRAM (GPU only) |
| `DEVICE_OVERRIDES` | `{}` | per-model device, e.g. `{"bge_m3":"cpu"}` |
| `CANDIDATE_MULTIPLIER` | `4` | per-model over-fetch before fusion |
| `RRF_K` | `60` | RRF damping |
| `MIN_MODEL_AGREEMENT` | `2` | models that must agree on the tag |
| `MIN_COSINE` | `0.78` | similarity floor below which nothing is answered |
| `CONTEXT_TURNS` | `2` | prior questions folded into a follow-up |
| `CONTEXT_TRIGGER_WORDS` | `6` | at or below this length, borrow context |
| `EMBED_BATCH_SIZE` | `16` | raise to 64 on GPU |
| `EAGER_LOAD_MODELS` | `false` | load weights at boot, not first query |
| `LOG_LEVEL` | `INFO` | |
| `LOG_FILE` | `logs/ec-vectordb.log` | empty string = console only |
| `LOG_MAX_BYTES` | `10485760` | 10 MiB; truncated past this |
| `LOG_BACKUP_COUNT` | `0` | `0` truncates; higher keeps N old files |

## Known constraints

**torch must be ≥ 2.6.** `BAAI/bge-m3` publishes only `pytorch_model.bin` and
no safetensors, and `transformers` refuses to `torch.load` a pickle checkpoint
on older torch (CVE-2025-32434). On the original torch 2.5.1 base image the
service died at startup with *"we now require users to upgrade torch to at
least v2.6"* — e5 and qwen3 loaded fine, because both ship safetensors.

The base image is therefore `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime`,
and torch is **not** in `[project.dependencies]` — it lives in the optional
`torch` extra (`>=2.6`), so the image inherits the base copy instead of
pulling a second ~2.5 GB of CUDA libraries from PyPI, while a workstation
gets it with `pip install -e ".[dev]"`.
`src.embedding.preflight()` reports the problem at boot instead of letting it
surface as a stack trace mid-load. Do not pin torch back without first
checking whether bge-m3 has started shipping safetensors.

CUDA 12.8 rather than 12.4 so one image covers newer server GPUs (Blackwell /
sm_120) as well as Ampere.

**HF Xet buffers downloads in RAM.** Fetching the ~2.2 GB e5 checkpoint pushed
container memory to ~2.1 GiB before flushing to disk. Harmless when models load
sequentially on a 15 GiB host; set `HF_HUB_DISABLE_XET=1` to stream straight to
disk if memory is tight.

## Operational notes

- **Ingest is the expensive part**, not query — it embeds 17,995 questions
  three times. Run it once per model or dataset change.
- **Changing a model means re-ingesting.** Old points still hold the old
  model's vectors under that name.
- **Credentials.** `data/` is fetched from a private repo. `scripts/fetch_dataset.sh`
  prefers an authenticated `gh` CLI (token stays in the OS keyring) and falls
  back to `GITHUB_TOKEN` from `.env`, which is gitignored. Prefer the keyring.
