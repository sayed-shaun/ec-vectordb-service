"""Fetch the corpus CSV from the private GitHub repo."""

from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path

from src.core.config import settings
from src.ec import load_questions

log = logging.getLogger(__name__)


def fetch_dataset(dest: Path) -> int:
    """Download the corpus to ``dest`` and return its row count.

    The download lands in a temp file and replaces ``dest`` only once it
    parses to at least one row, so a failed or truncated fetch never leaves
    the previous good copy replaced.
    """
    url = (
        f"https://api.github.com/repos/{settings.DATASET_REPO}/contents/"
        f"{settings.DATASET_PATH}?ref={settings.DATASET_REF}"
    )
    headers = {"Accept": "application/vnd.github.raw"}
    if settings.GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {settings.GITHUB_TOKEN}"

    log.info("fetching %s@%s:%s", settings.DATASET_REPO, settings.DATASET_REF, settings.DATASET_PATH)
    req = urllib.request.Request(url, headers=headers)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    try:
        with urllib.request.urlopen(req, timeout=settings.DATASET_TIMEOUT_S) as resp:
            tmp.write_bytes(resp.read())
        rows = len(load_questions(tmp))
        if not rows:
            raise ValueError("fetched dataset has no usable rows")
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
    return rows
