"""The compat normaliser must agree byte-for-byte with the reference service.

Every expectation below was captured from the live service at :8002; if one of
these fails, queries and the index have diverged and recall drops silently.
"""

import pytest

from src.normalize import normalize, normalize_compat, normalize_safe

OBSERVED = [
    ("নতুন ভোটার হতে কি কি লাগে?।", "নতুন ভোটার হতে কি কি লাগে"),
    ("  নতুন   ভোটার    হতে  ", "নতুন ভোটার হতে"),
    ("১৮ বছর বয়সে ভোটার", "১৮ বছর বয়সে ভোটার"),
    ("18 years NID card", "18 years nid card"),
    ("ভোটার, তালিকা: হালনাগাদ! (নতুন)", "ভোটার তালিকা হালনাগাদ নতুন"),
    ("ভোটার,তালিকা", "ভোটারতালিকা"),
    ("e-service NID-card", "eservice nidcard"),
    ("৫.৫ শতাংশ 3.14", "৫৫ শতাংশ 314"),
    ("a@b #tag c&d", "ab tag c&d"),
    ("৳১০০ $50", "১০০ $50"),
    ("১০০ % ভাগ", "১০০ % ভাগ"),
    ("ভোটার\tতালিকা\nহালনাগাদ", "ভোটার তালিকা হালনাগাদ"),
    ("ক‌খ‍গ", "কখগ"),
]


@pytest.mark.parametrize("raw,expected", OBSERVED)
def test_compat_matches_reference_service(raw, expected):
    assert normalize_compat(raw) == expected


def test_compat_preserves_the_reference_asymmetries():
    assert normalize_compat("“x”") == "x”"
    assert normalize_compat("a–b—c") == "ab—c"


def test_safe_normalizer_keeps_token_boundaries():
    assert normalize_compat("ভোটার,তালিকা") == "ভোটারতালিকা"
    assert normalize_safe("ভোটার,তালিকা") == "ভোটার তালিকা"


def test_safe_normalizer_applies_nfc():
    decomposed = "ড়"
    precomposed = "ড়"
    assert normalize_safe(decomposed) == normalize_safe(precomposed)


def test_none_normalizer_only_strips_outer_whitespace():
    assert normalize("  Hello, World?  ", "none") == "Hello, World?"


def test_unknown_normalizer_rejected():
    with pytest.raises(ValueError, match="unknown normalizer"):
        normalize("x", "bogus")
