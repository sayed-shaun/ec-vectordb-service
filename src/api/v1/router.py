"""Aggregates the v1 service routers into one mountable router.

`main` includes this and nothing else, so adding a service is a one-line edit
here rather than a change to the application module.
"""

from __future__ import annotations

from fastapi import APIRouter

from src.api.v1.topk.router import router as topk_router
from src.api.v1.reindex.router import router as reindex_router
from src.api.v1.system.router import router as system_router

router = APIRouter()
router.include_router(system_router)
router.include_router(topk_router)
router.include_router(reindex_router)
