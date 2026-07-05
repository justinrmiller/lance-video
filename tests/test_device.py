from __future__ import annotations

import pytest

from video_lance import device as device_mod
from video_lance.device import autodetect_device, resolve_device


def test_autodetect_returns_known_device() -> None:
    assert autodetect_device() in {"cuda", "mps", "cpu"}


def test_resolve_passthrough() -> None:
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"
    assert resolve_device("mps") == "mps"


def test_resolve_auto_uses_autodetect(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device_mod, "autodetect_device", lambda: "cuda")
    assert resolve_device("auto") == "cuda"


def test_resolve_unknown_rejected() -> None:
    with pytest.raises(ValueError):
        resolve_device("tpu")
