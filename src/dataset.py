"""Fetch the corpus CSV from the private GitHub repo."""

from __future__ import annotations

import logging
import os
import urllib.error
import urllib.request
from pathlib import Path

from src.core.config import settings
from src.ec import load_questions

log = logging.getLogger(__name__)


def fetch_dataset(dest: Path, token: str) -> int:
    """Download the corpus to ``dest`` with ``token`` and return its row count.

    The download lands in a temp file and replaces ``dest`` only once it
    parses to at least one row, so a failed or truncated fetch never leaves
    the previous good copy replaced.
    """
    url = (
        f"https://api.github.com/repos/{settings.DATASET_REPO}/contents/"
        f"{settings.DATASET_PATH}?ref={settings.DATASET_REF}"
    )
    headers = {
        "Accept": "application/vnd.github.raw",
        "Authorization": f"Bearer {token}",
    }

    log.info(
        "fetching %s@%s:%s",
        settings.DATASET_REPO, settings.DATASET_REF, settings.DATASET_PATH,
    )
    req = urllib.request.Request(url, headers=headers)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    try:
        try:
            with urllib.request.urlopen(req, timeout=settings.DATASET_TIMEOUT_S) as resp:
                tmp.write_bytes(resp.read())
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403, 404):
                # GitHub answers 404 as well as 401 to a caller it will not
                # show a private repo to. The token is never echoed back.
                raise RuntimeError(
                    f"HTTP {exc.code} fetching the dataset: the X-GitHub-Token was "
                    "rejected or cannot read the repo (scope, SSO, expiry)"
                ) from exc
            raise
        rows = len(load_questions(tmp))
        if not rows:
            raise ValueError("fetched dataset has no usable rows")
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
    return rows
