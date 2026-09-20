"""Query normalisation, bit-compatible with the EC bot service at :8002.

The compat strip set reproduces that service exactly, asymmetries included;
see the Normalisation section of the README.
"""

from __future__ import annotations

import re
import unicodedata

COMPAT_STRIP_CHARS = "!\"#'()*+,-./:=?@[]^`|~।॥৳“‘’–…‌‍﻿"

_COMPAT_STRIP_RE = re.compile(f"[{re.escape(COMPAT_STRIP_CHARS)}]")
_WHITESPACE_RE = re.compile(r"\s+")

_SAFE_STRIP_RE = re.compile(r"[^\w\sঀ-৿%]", flags=re.UNICODE)


def normalize_compat(text: str) -> str:
    """Reproduce the reference service's ``normalized_question`` exactly."""
    text = _COMPAT_STRIP_RE.sub("", text)
    text = text.lower()
    return _WHITESPACE_RE.sub(" ", text).strip()


def normalize_safe(text: str) -> str:
    """NFC-normalise, punctuation to spaces, lowercase, collapse whitespace.

    Preferred for a freshly built index; keeps token boundaries intact.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("‌", "").replace("‍", "").replace("﻿", "")
    text = _SAFE_STRIP_RE.sub(" ", text)
    text = text.lower()
    return _WHITESPACE_RE.sub(" ", text).strip()


NORMALIZERS = {
    "compat": normalize_compat,
    "safe": normalize_safe,
    "none": lambda t: t.strip(),
}


def normalize(text: str, mode: str = "compat") -> str:
    try:
        return NORMALIZERS[mode](text)
    except KeyError:
        raise ValueError(f"unknown normalizer {mode!r}; expected one of {list(NORMALIZERS)}")
