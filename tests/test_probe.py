from __future__ import annotations

from pathlib import Path

import pytest

from video_lance.probe import ProbeError, get_meta


def test_probe_fixture(fixture_video: Path) -> None:
    meta = get_meta(fixture_video)
    assert meta.duration_s == pytest.approx(10.0, abs=0.1)
    assert meta.width == 320
    assert meta.height == 240
    assert meta.fps == pytest.approx(30.0, abs=0.1)
    assert meta.codec == "h264"
    assert meta.size_bytes > 0
    assert meta.size_bytes == fixture_video.stat().st_size


def test_probe_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        get_meta(tmp_path / "nope.mp4")


def test_probe_non_video_raises(tmp_path: Path) -> None:
    bogus = tmp_path / "not_a_video.mp4"
    bogus.write_bytes(b"this is not a video")
    with pytest.raises(ProbeError):
        get_meta(bogus)
