"""Lazy-loaded embedding backends, one per configured model."""

from __future__ import annotations

import logging
import threading
from typing import Protocol

import torch
from sentence_transformers import SentenceTransformer

from src.core.config import MODEL_SPECS, ModelSpec, settings

log = logging.getLogger(__name__)


class Embedder(Protocol):
    spec: ModelSpec

    def embed_queries(self, texts: list[str]) -> torch.Tensor: ...

    def embed_passages(self, texts: list[str]) -> torch.Tensor: ...


class SentenceTransformerEmbedder:
    """Wraps a sentence-transformers model, loading it on first use.

    The load is locked: FastAPI serves from a thread pool, and concurrent
    first requests would otherwise each pull the weights into memory.
    """

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self._model = None
        self._lock = threading.Lock()

    def _ensure_loaded(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                device = self.spec.resolved_device()
                dtype = self.spec.resolved_dtype()
                log.info(
                    "loading %s (%s) on %s/%s (~%.0f MiB weights)",
                    self.spec.name, self.spec.hf_id, device, dtype,
                    self.spec.vram_estimate_mib(),
                )
                model_kwargs = {}
                if dtype != "float32" and device.startswith("cuda"):
                    model_kwargs["torch_dtype"] = getattr(torch, dtype)
                model = SentenceTransformer(
                    self.spec.hf_id,
                    device=device,
                    cache_folder=settings.MODEL_CACHE_DIR,
                    trust_remote_code=True,
                    model_kwargs=model_kwargs or None,
                )
                if self.spec.max_seq_length:
                    model.max_seq_length = self.spec.max_seq_length
                self._model = model
        return self._model

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _encode(self, texts: list[str]) -> torch.Tensor:
        if not texts:
            return torch.empty(0, self.spec.dim)
        return self._ensure_loaded().encode(
            texts,
            batch_size=settings.EMBED_BATCH_SIZE,
            normalize_embeddings=self.spec.normalize,
            convert_to_tensor=True,
            show_progress_bar=False,
        ).cpu()

    def embed_queries(self, texts: list[str]) -> torch.Tensor:
        return self._encode([self.spec.encode_query(t) for t in texts])

    def embed_passages(self, texts: list[str]) -> torch.Tensor:
        return self._encode([self.spec.encode_passage(t) for t in texts])


class EmbedderRegistry:
    """Holds one embedder per configured model."""

    def __init__(self, specs: tuple[ModelSpec, ...] = MODEL_SPECS) -> None:
        self._embedders: dict[str, Embedder] = {
            spec.name: SentenceTransformerEmbedder(spec) for spec in specs
        }

    @property
    def names(self) -> list[str]:
        return list(self._embedders)

    def register(self, embedder: Embedder) -> None:
        """Swap in an alternative backend for one model slot."""
        self._embedders[embedder.spec.name] = embedder

    def embed_queries(self, text: str, models: list[str] | None = None) -> dict[str, torch.Tensor]:
        targets = models or self.names
        return {name: self._embedders[name].embed_queries([text])[0] for name in targets}

    def embed_passages(self, texts: list[str]) -> dict[str, torch.Tensor]:
        return {name: emb.embed_passages(texts) for name, emb in self._embedders.items()}

    def warmup(self) -> None:
        for emb in self._embedders.values():
            emb.embed_queries(["warmup"])

    def status(self) -> dict[str, dict]:
        return {
            emb.spec.name: {
                "hf_id": emb.spec.hf_id,
                "dim": emb.spec.dim,
                "device": emb.spec.resolved_device(),
                "dtype": emb.spec.resolved_dtype(),
                "weight_mib": round(emb.spec.vram_estimate_mib(), 1),
                "weight": emb.spec.weight,
                "loaded": getattr(emb, "is_loaded", True),
            }
            for emb in self._embedders.values()
        }


registry = EmbedderRegistry()
