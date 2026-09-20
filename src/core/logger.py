"""Logging setup: a console stream plus a size-capped file.

The file is hard-capped at ``LOG_MAX_BYTES`` (10 MiB by default).
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.core.config import settings

FORMAT = "%(asctime)s %(levelname)-8s %(name)s %(message)s"
DATEFMT = "%Y-%m-%d %H:%M:%S"

_UVICORN_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")

_NOISY_LOGGERS = ("httpx", "httpcore", "urllib3", "sentence_transformers", "transformers")

_configured = False


class CappedFileHandler(RotatingFileHandler):
    """Rotating handler that truncates in place instead of rotating.

    ``backupCount == 0`` discards the file on rollover, capping disk use at
    ``maxBytes`` and losing the older history. Above 0 it is the stdlib
    handler unchanged. The stdlib alone does not cap: it reopens in append
    mode and grows without bound.
    """

    def doRollover(self) -> None:
        if self.backupCount > 0:
            super().doRollover()
            return

        if self.stream:
            self.stream.close()
            self.stream = None
        with open(self.baseFilename, "w", encoding=self.encoding or "utf-8"):
            pass
        if not self.delay:
            self.stream = self._open()
            self.stream.write(
                f"--- log truncated at {self.maxBytes} bytes; earlier lines discarded ---\n"
            )


def _build_file_handler(level: int) -> logging.Handler | None:
    if not settings.log_file:
        return None

    path = Path(settings.log_file)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = CappedFileHandler(
            path,
            maxBytes=settings.log_max_bytes,
            backupCount=settings.log_backup_count,
            encoding="utf-8",
        )
    except OSError as exc:
        logging.getLogger(__name__).warning("file logging disabled (%s): %s", path, exc)
        return None

    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(FORMAT, DATEFMT))
    return handler


def setup_logging(force: bool = False) -> logging.Logger:
    """Install the console and file handlers on the root logger. Idempotent."""
    global _configured
    root = logging.getLogger()
    if _configured and not force:
        return root

    level = logging.getLevelNamesMapping().get(settings.log_level.upper(), logging.INFO)

    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.setLevel(level)

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter(FORMAT, DATEFMT))
    root.addHandler(console)

    file_handler = _build_file_handler(level)
    if file_handler is not None:
        root.addHandler(file_handler)

    for name in _UVICORN_LOGGERS:
        uvicorn_log = logging.getLogger(name)
        uvicorn_log.handlers.clear()
        uvicorn_log.propagate = True

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(max(level, logging.WARNING))

    _configured = True
    if file_handler is not None:
        root.debug(
            "logging to %s (cap %.1f MiB, %d backups)",
            os.path.abspath(settings.log_file),
            settings.log_max_bytes / 2**20,
            settings.log_backup_count,
        )
    return root
