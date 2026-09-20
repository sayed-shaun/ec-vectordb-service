"""Test doubles: deterministic hash embedders and an in-memory Qdrant.

qdrant-client's ``:memory:`` mode runs the real query/filter logic in-process,
so these are integration tests of everything but the neural nets.
"""

from __future__ import annotations

import hashlib

import pytest
from qdrant_client import QdrantClient

from src.core.config import MODEL_SPECS
from src.embedding import EmbedderRegistry
from src.service import RetrievalService
from src.store import VectorStore


class FakeEmbedder:
    """Deterministic hash-bucket embedder.

    Each token lands in a fixed dimension, so shared vocabulary means high
    cosine similarity -- enough structure for ranking assertions. ``salt``
    makes each model disagree slightly, which is the whole point of voting.
    """

    def __init__(self, spec, salt: str = "") -> None:
        self.spec = spec
        self.salt = salt
        self.is_loaded = True

    def _vec(self, text: str) -> list[float]:
        vec = [0.0] * self.spec.dim
        for token in text.lower().split():
            digest = hashlib.sha256(f"{self.salt}{token}".encode()).digest()
            idx = int.from_bytes(digest[:4], "big") % self.spec.dim
            vec[idx] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_queries(self, texts):
        return [self._vec(self.spec.encode_query(t)) for t in texts]

    def embed_passages(self, texts):
        return [self._vec(self.spec.encode_passage(t)) for t in texts]


@pytest.fixture
def embedders() -> EmbedderRegistry:
    reg = EmbedderRegistry()
    for i, spec in enumerate(MODEL_SPECS):
        reg.register(FakeEmbedder(spec, salt=f"m{i}:"))
    return reg


@pytest.fixture
def store() -> VectorStore:
    vs = VectorStore(client=QdrantClient(":memory:"), collection="test_docs")
    vs.ensure_collection()
    return vs


@pytest.fixture
def service(store, embedders) -> RetrievalService:
    return RetrievalService(store=store, embedders=embedders)
