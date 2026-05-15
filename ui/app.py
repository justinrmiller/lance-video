"""Hugging Face Spaces entry point.

Spaces convention: a top-level `app.py` that exposes a Gradio `demo` and
calls `.launch()` when run as `__main__`. All the actual UI lives in
`video_lance.ui_app`; this shim only wires environment variables to the
launch arguments so the Space metadata can configure things like the
LanceDB path via env vars instead of code changes.

Required env (defaults shown):

    VL_DB_PATH   ./video-lance.db
    VL_DEVICE    auto
    VL_HOST      0.0.0.0       (Spaces sets this — bind on all interfaces)
    VL_PORT      7860
"""

from __future__ import annotations

import os
from pathlib import Path

from video_lance.ui_app import build_app, build_context

_db_path = Path(os.environ.get("VL_DB_PATH", "./video-lance.db"))
_device = os.environ.get("VL_DEVICE", "auto")

# Build at import time so HF Spaces' health check sees a ready Blocks object.
_ctx = build_context(_db_path, device=_device)
demo = build_app(_ctx)


if __name__ == "__main__":
    demo.launch(
        server_name=os.environ.get("VL_HOST", "0.0.0.0"),
        server_port=int(os.environ.get("VL_PORT", "7860")),
    )
