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
        f"https://api.github.com/repos/{settings.dataset_repo}/contents/"
        f"{settings.dataset_path}?ref={settings.dataset_ref}"
    )
    headers = {"Accept": "application/vnd.github.raw"}
    if settings.github_token:
        headers["Authorization"] = f"Bearer {settings.github_token}"

    log.info("fetching %s@%s:%s", settings.dataset_repo, settings.dataset_ref, settings.dataset_path)
    req = urllib.request.Request(url, headers=headers)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    try:
        with urllib.request.urlopen(req, timeout=settings.dataset_timeout_s) as resp:
            tmp.write_bytes(resp.read())
        rows = len(load_questions(tmp))
        if not rows:
            raise ValueError("fetched dataset has no usable rows")
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
    return rows
