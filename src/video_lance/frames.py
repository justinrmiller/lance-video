from __future__ import annotations

import io
import shutil
import subprocess
from pathlib import Path

from PIL import Image

from video_lance.config import FrameSamplingConfig


class FrameExtractError(RuntimeError):
    pass


def _ffmpeg_path() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise FrameExtractError(
            "ffmpeg not found on PATH; install ffmpeg (e.g. `brew install ffmpeg`)"
        )
    return path


def _downscale_long_edge(image: Image.Image, max_long_edge: int) -> Image.Image:
    long_edge = max(image.width, image.height)
    if long_edge <= max_long_edge:
        return image
    scale = max_long_edge / long_edge
    new_size = (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale))))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def extract_keyframe(
    path: Path,
    t_s: float,
    cfg: FrameSamplingConfig,
) -> tuple[bytes, Image.Image]:
    """Extract a single frame at `t_s` from `path`.

    Returns a (jpeg_bytes, PIL.Image) pair. The image is the post-resize RGB
    PIL image (useful for re-encoding into embedders without going back to
    disk); jpeg_bytes is the same image encoded as JPEG at
    `cfg.jpeg_quality` and downscaled to a long edge of `cfg.max_long_edge`.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    if t_s < 0:
        raise ValueError(f"t_s must be >= 0, got {t_s}")

    args = [
        _ffmpeg_path(),
        "-nostdin",
        "-loglevel",
        "error",
        "-ss",
        f"{t_s}",
        "-i",
        str(path),
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "pipe:1",
    ]
    result = subprocess.run(args, capture_output=True, check=False)
    if result.returncode != 0 or not result.stdout:
        raise FrameExtractError(
            f"ffmpeg frame extraction failed for {path} at t={t_s}: "
            f"{result.stderr.decode(errors='replace').strip()}"
        )

    image = Image.open(io.BytesIO(result.stdout)).convert("RGB")
    image = _downscale_long_edge(image, cfg.max_long_edge)

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=cfg.jpeg_quality)
    return buf.getvalue(), image
