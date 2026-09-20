"""The cap is the whole point of the custom handler, so it is tested directly.

The stdlib ``RotatingFileHandler`` with ``backupCount=0`` does *not* cap the
file -- it reopens in append mode after rollover -- which is the bug these
tests exist to pin.
"""

from __future__ import annotations

import logging

from src.core.logger import CappedFileHandler

CAP = 2048


def _handler(path, backups=0):
    handler = CappedFileHandler(path, maxBytes=CAP, backupCount=backups, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    return handler


def _emit(handler, count, payload="x" * 120):
    for i in range(count):
        handler.emit(
            logging.LogRecord("t", logging.INFO, __file__, i, f"{i:04d} {payload}", (), None)
        )


def test_file_never_exceeds_cap(tmp_path):
    path = tmp_path / "svc.log"
    handler = _handler(path)
    try:
        _emit(handler, 250)
    finally:
        handler.close()

    assert path.stat().st_size <= CAP
    assert sorted(p.name for p in tmp_path.iterdir()) == ["svc.log"], "no rotated siblings"


def test_truncation_keeps_the_newest_lines(tmp_path):
    path = tmp_path / "svc.log"
    handler = _handler(path)
    try:
        _emit(handler, 250)
    finally:
        handler.close()

    text = path.read_text(encoding="utf-8")
    assert "log truncated" in text, "the marker explains the gap to whoever reads the file"
    assert "0000 " not in text, "oldest lines are the ones discarded"
    assert text.rstrip().endswith("x" * 120), "the most recent line survived"


def test_backup_count_restores_stdlib_rotation(tmp_path):
    path = tmp_path / "svc.log"
    handler = _handler(path, backups=2)
    try:
        _emit(handler, 250)
    finally:
        handler.close()

    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["svc.log", "svc.log.1", "svc.log.2"]
    for name in names:
        assert (tmp_path / name).stat().st_size <= CAP


def test_setup_logging_is_idempotent(tmp_path, monkeypatch):
    from src.core import logger as logger_mod
    from src.core.config import settings

    monkeypatch.setattr(settings, "log_file", str(tmp_path / "svc.log"))
    monkeypatch.setattr(logger_mod, "_configured", False)

    root = logger_mod.setup_logging(force=True)
    first = list(root.handlers)
    logger_mod.setup_logging()
    assert list(root.handlers) == first, "a second call must not double every line"

    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()


def test_missing_log_dir_is_created(tmp_path, monkeypatch):
    from src.core import logger as logger_mod
    from src.core.config import settings

    target = tmp_path / "deep" / "nested" / "svc.log"
    monkeypatch.setattr(settings, "log_file", str(target))
    monkeypatch.setattr(logger_mod, "_configured", False)

    root = logger_mod.setup_logging(force=True)
    try:
        logging.getLogger("t").info("hello")
        assert target.exists()
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()
