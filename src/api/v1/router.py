"""Aggregates the v1 service routers into one mountable router.

`main` includes this and nothing else, so adding a service is a one-line edit
here rather than a change to the application module.
"""

from __future__ import annotations

from fastapi import APIRouter

from src.api.v1.ec_bot.router import router as ec_bot_router
from src.api.v1.qa.router import router as qa_router
from src.api.v1.system.router import router as system_router

router = APIRouter()
router.include_router(system_router)
router.include_router(ec_bot_router)
router.include_router(qa_router)
