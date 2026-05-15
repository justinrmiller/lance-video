from __future__ import annotations

VALID_DEVICES = frozenset({"auto", "cuda", "mps", "cpu"})


def autodetect_device() -> str:
    """Return the best available torch device: 'cuda' > 'mps' > 'cpu'."""
    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    backends_mps = getattr(torch.backends, "mps", None)
    if backends_mps is not None and backends_mps.is_available():
        return "mps"
    return "cpu"


def resolve_device(device: str) -> str:
    """Map 'auto' to the autodetected device; pass through 'cuda'/'mps'/'cpu'."""
    if device not in VALID_DEVICES:
        raise ValueError(
            f"unknown device {device!r}; expected one of {sorted(VALID_DEVICES)}"
        )
    if device == "auto":
        return autodetect_device()
    return device
