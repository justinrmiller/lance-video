from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from tests.fixtures.make_fixture import COLOR_RGB, color_at
from video_lance.config import FrameSamplingConfig
from video_lance.frames import FrameExtractError, extract_keyframe


def _center_pixel(image: Image.Image) -> tuple[int, int, int]:
    px = image.getpixel((image.width // 2, image.height // 2))
    assert isinstance(px, tuple) and len(px) == 3
    return px


def _approx_equal(a: tuple[int, int, int], b: tuple[int, int, int], tol: int = 8) -> bool:
    return all(abs(int(x) - int(y)) <= tol for x, y in zip(a, b, strict=True))


@pytest.mark.parametrize("t_s", [1.0, 3.0, 5.0, 7.0, 9.0])
def test_frame_color_at_known_times(fixture_video: Path, t_s: float) -> None:
    cfg = FrameSamplingConfig()
    _, image = extract_keyframe(fixture_video, t_s, cfg)
    expected = COLOR_RGB[color_at(t_s)]
    actual = _center_pixel(image)
    assert _approx_equal(actual, expected), f"t={t_s}: got {actual}, expected ~{expected}"


def test_frame_jpeg_decodes(fixture_video: Path) -> None:
    cfg = FrameSamplingConfig()
    jpeg_bytes, _ = extract_keyframe(fixture_video, 1.0, cfg)
    decoded = Image.open(io.BytesIO(jpeg_bytes))
    decoded.verify()


def test_frame_downscaled_to_max_long_edge(fixture_video: Path) -> None:
    cfg = FrameSamplingConfig(max_long_edge=160)
    _, image = extract_keyframe(fixture_video, 1.0, cfg)
    # Source is 320x240; long edge 320 → halved to 160; short edge → 120.
    assert max(image.width, image.height) == 160
    assert image.width == 160
    assert image.height == 120


def test_frame_no_upscale_when_below_threshold(fixture_video: Path) -> None:
    cfg = FrameSamplingConfig(max_long_edge=512)
    _, image = extract_keyframe(fixture_video, 1.0, cfg)
    # Source is 320x240 and threshold is 512 — should be untouched.
    assert image.width == 320
    assert image.height == 240


def test_frame_jpeg_quality_affects_size(fixture_video: Path) -> None:
    low = FrameSamplingConfig(jpeg_quality=20)
    high = FrameSamplingConfig(jpeg_quality=95)
    low_bytes, _ = extract_keyframe(fixture_video, 1.0, low)
    high_bytes, _ = extract_keyframe(fixture_video, 1.0, high)
    assert len(high_bytes) > len(low_bytes)


def test_frame_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        extract_keyframe(tmp_path / "nope.mp4", 1.0, FrameSamplingConfig())


def test_frame_negative_time_rejected(fixture_video: Path) -> None:
    with pytest.raises(ValueError):
        extract_keyframe(fixture_video, -1.0, FrameSamplingConfig())


def test_frame_bad_video_raises(tmp_path: Path) -> None:
    bogus = tmp_path / "bad.mp4"
    bogus.write_bytes(b"not a video")
    with pytest.raises(FrameExtractError):
        extract_keyframe(bogus, 0.5, FrameSamplingConfig())
