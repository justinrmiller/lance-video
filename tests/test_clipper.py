from __future__ import annotations

from pathlib import Path

import pytest

from video_lance.clipper import (
    _ACCURATE_SEEK_WINDOW,
    ClipError,
    _accurate_seek_args,
    extract_clip_bytes,
)
from video_lance.probe import get_meta


def _write_and_probe(tmp_path: Path, data: bytes, name: str = "clip.mp4") -> tuple[Path, object]:
    out = tmp_path / name
    out.write_bytes(data)
    return out, get_meta(out)


def test_clip_extraction_basic(fixture_video: Path, tmp_path: Path) -> None:
    data = extract_clip_bytes(fixture_video, 2.0, 6.0)
    assert len(data) > 0
    # ftyp/moov header marker — any valid MP4 starts with an ftyp atom early on.
    assert b"ftyp" in data[:64]
    _, meta = _write_and_probe(tmp_path, data)
    assert meta.duration_s == pytest.approx(4.0, abs=0.5)


def test_clip_extraction_precise(fixture_video: Path, tmp_path: Path) -> None:
    data = extract_clip_bytes(fixture_video, 1.0, 4.0, precise=True)
    assert len(data) > 0
    _, meta = _write_and_probe(tmp_path, data, "precise.mp4")
    assert meta.duration_s == pytest.approx(3.0, abs=0.5)


def test_accurate_seek_args_small_offset_output_seeks() -> None:
    # Small start: output seek after -i (frame-accurate): -i PATH -ss start.
    args = _accurate_seek_args(Path("/x.mp4"), 2.0)
    assert args == ["-i", "/x.mp4", "-ss", "2.0"]
    assert args.index("-ss") > args.index("-i")


def test_accurate_seek_args_large_offset_two_stage() -> None:
    start = _ACCURATE_SEEK_WINDOW + 25.0
    args = _accurate_seek_args(Path("/x.mp4"), start)
    # Layout: ["-ss", coarse, "-i", PATH, "-ss", fine]
    assert args[0] == "-ss"
    i_idx = args.index("-i")
    coarse = float(args[1])
    assert args[i_idx + 2] == "-ss"  # fine seek follows -i (accurate)
    fine = float(args[i_idx + 3])
    assert coarse == pytest.approx(start - _ACCURATE_SEEK_WINDOW)
    assert coarse + fine == pytest.approx(start)


def test_clip_rejects_invalid_window(fixture_video: Path) -> None:
    with pytest.raises(ValueError):
        extract_clip_bytes(fixture_video, 5.0, 5.0)
    with pytest.raises(ValueError):
        extract_clip_bytes(fixture_video, 5.0, 4.0)


def test_clip_missing_input(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        extract_clip_bytes(tmp_path / "nope.mp4", 0.0, 2.0)


def test_clip_bad_input_raises(tmp_path: Path) -> None:
    bogus = tmp_path / "bad.mp4"
    bogus.write_bytes(b"not a video")
    with pytest.raises(ClipError):
        extract_clip_bytes(bogus, 0.0, 1.0)
