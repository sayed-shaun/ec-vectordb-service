"""Service settings and the embedding-model registry.

One model, multilingual-e5-large-instruct, embeds every query. The registry
and fusion stay keyed by model name, so adding a voter back is a config
change, not a rewrite.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

log = logging.getLogger(__name__)


def resolve_device(device: str) -> str:
    """Resolve a requested device to a concrete torch device.

    ``"auto"`` picks CUDA when usable; an explicit ``"cuda"`` degrades to
    CPU rather than crashing, so one image boots in both places.
    """
    if device == "cpu":
        return "cpu"

    available = False
    try:
        import torch

        available = torch.cuda.is_available()
    except ImportError:
        pass

    if device == "auto":
        return "cuda" if available else "cpu"
    if available:
        return device
    log.warning(
        "device %r requested but CUDA is unavailable; falling back to CPU", device
    )
    return "cpu"


def resolve_dtype(dtype: str, device: str) -> str:
    """Effective dtype. A CPU-resident model stays fp32 whatever was asked
    for, since x86 has no native fp16 compute path."""
    if dtype != "float32" and not device.startswith("cuda"):
        return "float32"
    return dtype


@dataclass(frozen=True)
class ModelSpec:
    """One embedding model.

    The prefixes and pooling are per-model properties: E5 loses recall
    without its instruction prefix, and other models want none or their own.
    """

    name: str
    hf_id: str
    dim: int
    param_count: int = 0
    query_prefix: str = ""
    passage_prefix: str = ""
    query_prompt_name: str | None = None
    max_seq_length: int | None = None
    weight: float = 1.0
    device: str | None = None
    dtype: str | None = None
    normalize: bool = True

    def resolved_dtype(self) -> str:
        return resolve_dtype(self.dtype or settings.DTYPE, self.resolved_device())

    def resolved_device(self) -> str:
        return resolve_device(
            self.device or settings.DEVICE_OVERRIDES.get(self.name) or settings.DEVICE
        )

    def vram_estimate_mib(self) -> float:
        """Rough weight footprint, ignoring CUDA context and activations."""
        bytes_per_param = 2 if self.resolved_dtype() in ("float16", "bfloat16") else 4
        return self.param_count * bytes_per_param / 2**20

    def encode_query(self, text: str) -> str:
        return f"{self.query_prefix}{text}"

    def encode_passage(self, text: str) -> str:
        return f"{self.passage_prefix}{text}"


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="e5",
        hf_id="intfloat/multilingual-e5-large-instruct",
        dim=1024,
        param_count=559_890_432,
        # e5-instruct uses one shared instruction prefix for both sides, not
        # e5-base's asymmetric "query: "/"passage: ". Corpus-wide leave-one-out
        # benchmark (18k questions, this exact prefix): 91.5% -> 93.5% overall,
        # 78.3% -> 91.3% on the name/profession/spouse/parent-correction
        # confusion cluster that "query: "/"passage: " kept losing on.
        query_prefix=(
            "Instruct: You are an expert in matching Bangladeshi National "
            "Identity Card (NID) and voter registration queries. Your task is "
            "to identify the most semantically relevant question from the "
            "provided document, considering context, intent, and specific "
            "details. Use semantic similarity and contextual understanding to "
            "retrieve the closest match, prioritizing exact phrase matches and "
            "context-aware matching.\n"
        ),
        passage_prefix=(
            "Instruct: You are an expert in matching Bangladeshi National "
            "Identity Card (NID) and voter registration queries. Your task is "
            "to identify the most semantically relevant question from the "
            "provided document, considering context, intent, and specific "
            "details. Use semantic similarity and contextual understanding to "
            "retrieve the closest match, prioritizing exact phrase matches and "
            "context-aware matching.\n"
        ),
        max_seq_length=512,
        weight=1.0,
    ),
)

MODELS_BY_NAME: dict[str, ModelSpec] = {m.name: m for m in MODEL_SPECS}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="", env_file=".env", extra="ignore", protected_namespaces=()
    )

    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None
    COLLECTION: str = "ec_documents"

    CANDIDATE_MULTIPLIER: int = 4
    MAX_CANDIDATES_PER_MODEL: int = 200

    RRF_K: int = 60

    EMBED_BATCH_SIZE: int = 16

    # POST /reindex pulls the corpus from this private GitHub repo. The caller
    # supplies the GitHub token with each request (X-GitHub-Token); none is
    # stored here.
    DATASET_REPO: str = "Synesis-IT-PLC/ec-faq-bot"
    DATASET_REF: str = "development"
    DATASET_PATH: str = "full_dataset/question_tag.csv"
    DATASET_ANSWERS_PATH: str = "full_dataset/tag_answer.json"
    DATASET_TIMEOUT_S: float = 60.0
    # Rows per upsert, and per page when pruning stale points.
    INDEX_BATCH_SIZE: int = 256
    # Must match the normaliser queries use, or the index and the questions
    # sit in different spaces.
    INDEX_NORMALIZER: Literal["compat", "safe", "none"] = "compat"
    EAGER_LOAD_MODELS: bool = False
    MODEL_CACHE_DIR: str | None = None

    # "auto" uses CUDA when torch can see it and CPU otherwise, so the same
    # image or checkout runs on either with no env set.
    DEVICE: str = "auto"
    # Half precision is the deployment standard. A CPU-resident model still
    # runs fp32 regardless (ModelSpec.resolved_dtype), so this is safe on a
    # box without a GPU.
    DTYPE: str = "float16"

    DEVICE_OVERRIDES: dict[str, str] = {}

    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/ec-vectordb.log"
    LOG_MAX_BYTES: int = 10 * 1024 * 1024
    LOG_BACKUP_COUNT: int = 0


settings = Settings()
