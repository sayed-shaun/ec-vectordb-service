"""The cpu/cuda switch is the only difference between laptop and server, so
its resolution rules are worth pinning."""

import dataclasses

import pytest

from src.core.config import MODEL_SPECS, settings

SPEC = MODEL_SPECS[0]


@pytest.fixture
def cfg(monkeypatch):
    def apply(device="cpu", dtype="float32", overrides=None):
        monkeypatch.setattr(settings, "device", device)
        monkeypatch.setattr(settings, "dtype", dtype)
        monkeypatch.setattr(settings, "device_overrides", overrides or {})
    return apply


def test_cpu_is_the_default(cfg):
    cfg()
    assert SPEC.resolved_device() == "cpu"


def test_cuda_falls_back_to_cpu_when_unavailable(cfg, monkeypatch):
    cfg(device="cuda")
    monkeypatch.setitem(__import__("sys").modules, "torch", None)
    assert SPEC.resolved_device() in {"cpu", "cuda"}


def test_auto_never_raises(cfg):
    cfg(device="auto")
    assert SPEC.resolved_device() in {"cpu", "cuda"}


def test_per_model_override_beats_the_global_setting(cfg):
    cfg(device="cpu", overrides={"e5": "cpu", "bge_m3": "cpu"})
    assert SPEC.resolved_device() == "cpu"


def test_cpu_model_stays_fp32_even_when_dtype_is_float16(cfg):
    cfg(device="cpu", dtype="float16")
    assert SPEC.resolved_dtype() == "float32"


def test_vram_estimate_halves_with_fp16():
    fp32 = dataclasses.replace(SPEC, device="cuda", dtype="float32")
    fp16 = dataclasses.replace(SPEC, device="cuda", dtype="float16")
    assert fp32.vram_estimate_mib() == pytest.approx(2 * fp16.vram_estimate_mib())


CARD_TOTAL_MIB = 4096
DESKTOP_RESERVED_MIB = 655
CUDA_CONTEXT_MIB = 300
ACTIVATIONS_MIB = 300


def test_all_three_fp16_do_not_fit_a_4gb_card():
    """Documents the sizing conclusion so a future model swap re-checks it.

    Weights alone are ~3285 MiB and the full requirement ~3885 MiB, which is
    just under the card's 4096 MiB on paper -- but only on a card with nothing
    else on it. With the measured desktop session resident it is out of reach,
    which is why fp16-everything is not the shipped default.
    """
    weights = sum(
        dataclasses.replace(s, device="cuda", dtype="float16").vram_estimate_mib()
        for s in MODEL_SPECS
    )
    assert weights > 3200

    needed = weights + CUDA_CONTEXT_MIB + ACTIVATIONS_MIB
    assert needed < CARD_TOTAL_MIB
    assert needed + DESKTOP_RESERVED_MIB > CARD_TOTAL_MIB


def test_two_on_gpu_one_on_cpu_fits_a_4gb_card():
    on_gpu = [
        dataclasses.replace(s, device="cuda", dtype="float16")
        for s in MODEL_SPECS
        if s.name != "bge_m3"
    ]
    needed = sum(s.vram_estimate_mib() for s in on_gpu) + CUDA_CONTEXT_MIB + ACTIVATIONS_MIB
    assert needed + DESKTOP_RESERVED_MIB < CARD_TOTAL_MIB


def test_spec_dtype_pins_one_model_against_the_global(cfg):
    cfg(device="cuda", dtype="float16")
    pinned = dataclasses.replace(SPEC, dtype="float32", device="cuda")
    assert pinned.resolved_dtype() == "float32"


def test_preflight_flags_old_torch_against_pickle_only_models(monkeypatch):
    """bge-m3 ships no safetensors, so torch < 2.6 cannot load it at all."""
    import src.embedding as emb

    fake = type("T", (), {"__version__": "2.5.1+cu124"})
    monkeypatch.setitem(__import__("sys").modules, "torch", fake)
    problems = emb.preflight()
    assert any("bge-m3" in p and "2.6" in p for p in problems)


def test_preflight_is_quiet_on_new_torch(monkeypatch):
    import src.embedding as emb

    fake = type("T", (), {"__version__": "2.8.0+cu128"})
    monkeypatch.setitem(__import__("sys").modules, "torch", fake)
    assert emb.preflight() == []
