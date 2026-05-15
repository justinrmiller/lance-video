from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from video_lance.models import VideoMeta


class ProbeError(RuntimeError):
    pass


def _ffprobe_path() -> str:
    path = shutil.which("ffprobe")
    if not path:
        raise ProbeError("ffprobe not found on PATH; install ffmpeg (e.g. `brew install ffmpeg`)")
    return path


def _parse_fps(rate_str: str) -> float:
    if "/" in rate_str:
        num_s, den_s = rate_str.split("/", 1)
        num = float(num_s)
        den = float(den_s)
        if den == 0.0:
            return 0.0
        return num / den
    return float(rate_str)


def get_meta(path: Path) -> VideoMeta:
    if not path.exists():
        raise FileNotFoundError(path)

    cmd = [
        _ffprobe_path(),
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise ProbeError(f"ffprobe failed for {path}: {result.stderr.strip()}")

    data: dict[str, Any] = json.loads(result.stdout)
    fmt = data.get("format", {})
    streams = data.get("streams", [])

    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video_stream is None:
        raise ProbeError(f"no video stream in {path}")

    duration_s = float(fmt.get("duration") or video_stream.get("duration") or 0.0)
    fps = _parse_fps(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or "0")
    width = int(video_stream.get("width") or 0)
    height = int(video_stream.get("height") or 0)
    codec = str(video_stream.get("codec_name") or "")
    size_bytes = int(fmt.get("size") or path.stat().st_size)

    return VideoMeta(
        duration_s=duration_s,
        fps=fps,
        width=width,
        height=height,
        codec=codec,
        size_bytes=size_bytes,
    )
