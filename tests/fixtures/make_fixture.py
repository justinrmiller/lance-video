"""Generate a deterministic 10-second test video.

Layout:
- 320x240, 30 fps, H.264.
- Five 2-second color segments (red, green, blue, yellow, magenta) in order,
  so a frame at t=1.0 is solid red, t=3.0 is solid green, etc.
- 440 Hz sine audio for the full 10s (AAC).

The fixture is deterministic from these inputs; the test suite uses it for
probe + clip + frame extraction.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

WIDTH = 320
HEIGHT = 240
FPS = 30
SEGMENT_SECONDS = 2.0

# Order matters: the test suite relies on these being at the expected times.
COLORS: list[str] = ["red", "green", "blue", "yellow", "magenta"]

# Approximate sRGB tuples for the color names above. Tests sample pixels and
# compare against these; tolerance is generous because of YUV round-trip drift.
COLOR_RGB: dict[str, tuple[int, int, int]] = {
    "red": (255, 0, 0),
    "green": (0, 128, 0),  # ffmpeg's `color=green` is the SVG green, not lime
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "magenta": (255, 0, 255),
}


def color_at(t_s: float) -> str:
    """Return the color name expected at time `t_s` in the fixture."""
    if t_s < 0:
        raise ValueError(t_s)
    idx = int(t_s // SEGMENT_SECONDS)
    if idx >= len(COLORS):
        idx = len(COLORS) - 1
    return COLORS[idx]


def make_fixture(out_path: Path) -> Path:
    """(Re)generate the fixture video at `out_path`. Returns `out_path`."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH; cannot generate fixture")

    inputs: list[str] = []
    for color in COLORS:
        inputs.extend(
            [
                "-f",
                "lavfi",
                "-i",
                f"color=c={color}:size={WIDTH}x{HEIGHT}:rate={FPS}:duration={SEGMENT_SECONDS}",
            ]
        )
    total = SEGMENT_SECONDS * len(COLORS)
    inputs.extend(["-f", "lavfi", "-i", f"sine=frequency=440:duration={total}"])

    concat_inputs = "".join(f"[{i}:v]" for i in range(len(COLORS)))
    filter_complex = f"{concat_inputs}concat=n={len(COLORS)}:v=1:a=0[v]"

    cmd = [
        ffmpeg,
        "-nostdin",
        "-loglevel",
        "error",
        "-y",
        *inputs,
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-map",
        f"{len(COLORS)}:a",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-g",
        "30",
        "-c:a",
        "aac",
        "-shortest",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to build fixture: {result.stderr.decode(errors='replace')}"
        )
    return out_path


if __name__ == "__main__":
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/fixtures/sample.mp4")
    make_fixture(target)
    print(f"wrote {target}")
