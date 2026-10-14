"""Fetch the corpus files from the private GitHub repo."""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from src.core.config import settings
from src.ec import load_questions

log = logging.getLogger(__name__)


def _download(repo_path: str, dest: Path, token: str, count_rows: Callable[[Path], int]) -> int:
    """Download ``repo_path`` to ``dest`` with ``token`` and return its row count.

    The download lands in a temp file and replaces ``dest`` only once
    ``count_rows`` finds at least one row in it, so a failed or truncated
    fetch never replaces the previous good copy.
    """
    url = (
        f"https://api.github.com/repos/{settings.DATASET_REPO}/contents/"
        f"{repo_path}?ref={settings.DATASET_REF}"
    )
    headers = {
        "Accept": "application/vnd.github.raw",
        "Authorization": f"Bearer {token}",
    }

    log.info("fetching %s@%s:%s", settings.DATASET_REPO, settings.DATASET_REF, repo_path)
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
                    f"HTTP {exc.code} fetching {repo_path}: the X-GitHub-Token was "
                    "rejected or cannot read the repo (scope, SSO, expiry)"
                ) from exc
            raise
        rows = count_rows(tmp)
        if not rows:
            raise ValueError(f"fetched {repo_path} has no usable rows")
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
    return rows


def fetch_dataset(dest: Path, token: str) -> int:
    """The question/tag CSV; returns its row count."""
    return _download(
        settings.DATASET_PATH, dest, token, lambda p: len(load_questions(p))
    )


def _count_answers(path: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("tag_answer.json must map tag to answer")
    return len(data)


def fetch_answers(dest: Path, token: str) -> int:
    """The tag -> answer JSON; returns its entry count."""
    return _download(settings.DATASET_ANSWERS_PATH, dest, token, _count_answers)
