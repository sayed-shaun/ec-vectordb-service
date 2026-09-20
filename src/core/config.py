"""Service settings and the embedding-model registry.

Three models vote on every query, chosen to be genuinely different from each
other on Bangla text. See the README for why each one is here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from pydantic_settings import BaseSettings, SettingsConfigDict

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    """One voting member.

    The prefixes and pooling are per-model properties: E5 loses recall
    without its asymmetric prefixes, bge-m3 was trained without any.
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
        """Effective dtype. A CPU-resident model stays fp32 whatever the
        global setting says, since x86 has no native fp16 compute path."""
        dtype = self.dtype or settings.dtype
        if dtype != "float32" and not self.resolved_device().startswith("cuda"):
            return "float32"
        return dtype

    def resolved_device(self) -> str:
        """Resolve to a concrete torch device.

        ``"auto"`` picks CUDA when usable; an explicit ``"cuda"`` degrades to
        CPU rather than crashing, so one image boots in both places.
        """
        device = self.device or settings.device_overrides.get(self.name) or settings.device
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
        log.warning("device %r requested but CUDA is unavailable; falling back to CPU", device)
        return "cpu"

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
        hf_id="intfloat/multilingual-e5-large",
        dim=1024,
        param_count=559_890_432,
        query_prefix="query: ",
        passage_prefix="passage: ",
        max_seq_length=512,
        weight=1.0,
    ),
    ModelSpec(
        name="qwen3",
        hf_id="Qwen/Qwen3-Embedding-0.6B",
        dim=1024,
        param_count=595_776_512,
        query_prompt_name="query",
        max_seq_length=1024,
        weight=1.0,
    ),
    ModelSpec(
        name="bge_m3",
        hf_id="BAAI/bge-m3",
        dim=1024,
        param_count=566_393_856,
        max_seq_length=1024,
        weight=1.0,
    ),
)

MODELS_BY_NAME: dict[str, ModelSpec] = {m.name: m for m in MODEL_SPECS}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="", env_file=".env", extra="ignore", protected_namespaces=()
    )

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    collection: str = "ec_documents"

    candidate_multiplier: int = 4
    max_candidates_per_model: int = 200

    rrf_k: int = 60

    min_model_agreement: int = 2
    min_cosine: float = 0.78
    # An answer at or below this length is a placeholder, not an answer.
    # tag_answer.json ships `fraction` -> "tag" (3 chars, 772 questions); the
    # shortest real answer is 8, so this separates them with room to spare.
    min_answer_chars: int = 4

    # A follow-up like "how much?" names no subject on its own. An
    # external service rewrites it into a standalone question given the
    # conversation so far; empty disables the call and the raw question is
    # used unchanged, which keeps local dev and tests independent of it.
    rewrite_url: str = ""
    rewrite_timeout_s: float = 3.0

    # A chat_id lets the server hold a conversation's turns instead of the
    # caller echoing `history` back each time. In memory, per-process --
    # see src/core/memory.py for why, and what that costs.
    session_ttl_s: float = 1800.0       # idle this long and a chat is forgotten
    session_max_count: int = 10_000     # bound on concurrently held chats

    # The /ui test console. Off in production: it is a developer tool, and the
    # service has no authentication in front of it.
    enable_ui: bool = True
    embed_batch_size: int = 16

    eager_load_models: bool = False
    model_cache_dir: str | None = None

    device: str = "cpu"
    dtype: str = "float32"

    device_overrides: dict[str, str] = {}

    log_level: str = "INFO"
    log_file: str = "logs/ec-vectordb.log"
    log_max_bytes: int = 10 * 1024 * 1024
    log_backup_count: int = 0


settings = Settings()
