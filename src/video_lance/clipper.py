from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


class ClipError(RuntimeError):
    pass


def _ffmpeg_path() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise ClipError("ffmpeg not found on PATH; install ffmpeg (e.g. `brew install ffmpeg`)")
    return path


def _run_ffmpeg(args: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(args, capture_output=True, check=False)


# For a re-encode we can cut on an exact timestamp (not just a keyframe). Below
# this many seconds we output-seek from the start (accurate, cheap); beyond it
# we coarse input-seek to `start_s - _ACCURATE_SEEK_WINDOW` then fine output-seek
# the remainder — accurate and fast even far into a long file.
_ACCURATE_SEEK_WINDOW = 10.0


def _accurate_seek_args(path: Path, start_s: float) -> list[str]:
    """Frame-accurate seek args for a re-encode starting at `start_s`.

    The fine seek is an *output* seek (after `-i`), so the encoded clip begins
    exactly at `start_s` rather than at the nearest preceding keyframe. A `-t`
    duration placed after these args is measured from that exact start.
    """
    if start_s <= _ACCURATE_SEEK_WINDOW:
        return ["-i", str(path), "-ss", f"{start_s}"]
    coarse = start_s - _ACCURATE_SEEK_WINDOW
    return ["-ss", f"{coarse}", "-i", str(path), "-ss", f"{_ACCURATE_SEEK_WINDOW}"]


def extract_clip_bytes(
    path: Path,
    start_s: float,
    end_s: float,
    *,
    precise: bool = False,
) -> bytes:
    """Extract a clip from `path` covering [start_s, end_s) and return MP4 bytes.

    By default attempts stream copy (`-c copy`) for speed; falls back to a
    re-encode if stream copy fails or produces an empty file. Stream copy can
    only cut on keyframes, so the clip may begin at the nearest keyframe *before*
    `start_s` — fine when exact alignment doesn't matter.

    When `precise=True` we skip the stream-copy path entirely and re-encode with
    frame-accurate seeking (see `_accurate_seek_args`) so the clip starts exactly
    at `start_s` — this is what the ingest pipeline uses so a stored clip lines up
    with the transcript text mapped to the same window.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    if end_s <= start_s:
        raise ValueError(f"end_s ({end_s}) must be > start_s ({start_s})")

    ffmpeg = _ffmpeg_path()
    duration = end_s - start_s

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "clip.mp4"

        if not precise:
            copy_args = [
                ffmpeg,
                "-nostdin",
                "-loglevel",
                "error",
                "-ss",
                f"{start_s}",
                "-i",
                str(path),
                "-t",
                f"{duration}",
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                "-y",
                str(out),
            ]
            result = _run_ffmpeg(copy_args)
            if result.returncode == 0 and out.exists() and out.stat().st_size > 0:
                return out.read_bytes()

        encode_args = [
            ffmpeg,
            "-nostdin",
            "-loglevel",
            "error",
            *_accurate_seek_args(path, start_s),
            "-t",
            f"{duration}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            "-y",
            str(out),
        ]
        result = _run_ffmpeg(encode_args)
        if result.returncode != 0 or not out.exists() or out.stat().st_size == 0:
            raise ClipError(
                f"ffmpeg clip extraction failed for {path} "
                f"[{start_s}, {end_s}): {result.stderr.decode(errors='replace').strip()}"
            )
        return out.read_bytes()
