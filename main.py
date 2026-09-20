"""ASGI entrypoint: `uvicorn main:app`.

Only assembly lives here -- lifespan wiring and router mounting. The endpoints
themselves are in src/api/v1/<service>/router.py.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.deps import set_services
from src.api.v1.router import router as v1_router
from src.core.config import settings
from src.core.logger import setup_logging
from src.ec import AnswerStore
from src.ec_service import EcBotService
from src.embedding import preflight, registry
from src.service import RetrievalService
from src.store import VectorStore

setup_logging()
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = VectorStore()
    store.ensure_collection()
    retrieval = RetrievalService(store)

    ec_bot: EcBotService | None = None
    try:
        ec_bot = EcBotService(retrieval, AnswerStore.load())
        log.info("loaded %d tag answers", len(ec_bot.answers))
    except FileNotFoundError:
        log.warning("data/tag_answer.json not found; /ec_bot routes disabled")

    set_services(retrieval, ec_bot)

    for problem in preflight():
        log.error("PREFLIGHT: %s", problem)

    if settings.eager_load_models:
        log.info("warming up embedding models")
        registry.warmup()
    yield
    set_services(None, None)


app = FastAPI(
    title="EC VectorDB Service",
    version="0.1.0",
    description="Multi-model Bangla retrieval: three embedding models vote, RRF fuses.",
    lifespan=lifespan,
)

app.include_router(v1_router)
