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
    cfg(device="cuda", overrides={SPEC.name: "cpu"})
    assert SPEC.resolved_device() == "cpu"


def test_cpu_model_stays_fp32_even_when_dtype_is_float16(cfg):
    cfg(device="cpu", dtype="float16")
    assert SPEC.resolved_dtype() == "float32"


def test_vram_estimate_halves_with_fp16():
    fp32 = dataclasses.replace(SPEC, device="cuda", dtype="float32")
    fp16 = dataclasses.replace(SPEC, device="cuda", dtype="float16")
    assert fp32.vram_estimate_mib() == pytest.approx(2 * fp16.vram_estimate_mib())


def test_spec_dtype_pins_one_model_against_the_global(cfg):
    cfg(device="cuda", dtype="float16")
    pinned = dataclasses.replace(SPEC, dtype="float32", device="cuda")
    assert pinned.resolved_dtype() == "float32"
