# CLAUDE.md

Guidance for Claude Code (and other agents) working in this repository.

## What this is

`video-lance` indexes a directory of videos into a **LanceDB** store with
multi-modal embeddings and searches it from a CLI or a Gradio web app. Ingest
splits each video into overlapping time **segments** and stores, per segment: the
transcript text, a text embedding (e5-instruct, 1024-d), a keyframe's visual
embedding (SigLIP 2, 1152-d), the keyframe JPEG, and a short MP4 clip.

Deep-dive docs live in [`docs/`](docs/README.md): `architecture.md`,
`pipeline.md`, `search.md`, `cli-and-ui.md`, `development.md`.

## Commands

```bash
uv sync --all-groups                 # install runtime + dev deps
uv run pytest                        # tests (240 passed, 6 skipped, ~30s)
uv run ruff check src tests          # lint
uv run ruff format --check src tests scripts
uv run ty check                      # static types (Astral's `ty`, not mypy)
VL_INTEGRATION=1 uv run pytest tests/test_integration_real_models.py   # opt-in, downloads ~6GB

uv run video-lance ingest ./videos --db-path ./video-lance.db
uv run video-lance search "..." --mode text|visual|multi
uv run video-lance info | reindex
uv run video-lance ui                # Gradio at http://127.0.0.1:7860
```

Run the full gate (`pytest` + `ruff check` + `ruff format --check` + `ty check`)
before considering a change done.

## Requirements

- **Python 3.12+**, **ffmpeg + ffprobe** on `PATH`, **uv** for env management.
- Optional GPU (CUDA/MPS) — auto-detected, falls back to CPU.

## Architecture at a glance

Source is `src/video_lance/`. Rough layering (leaf → orchestration → front ends):

- **Pure / config**: `config.py`, `models.py`, `schema.py`, `device.py`,
  `discovery.py`, `segmenter.py`.
- **ffmpeg wrappers**: `probe.py`, `clipper.py`, `frames.py`.
- **Models (lazy, process-cached)**: `transcribe.py`, `embed_text.py`,
  `embed_vision.py`.
- **Storage**: `store.py` (LanceDB, three tables: `videos` / `segments` /
  `_metadata`).
- **Orchestration**: `stages.py` (8-stage `Stage` protocol), `pipeline.py`.
- **Retrieval**: `search.py`. **Front ends**: `cli.py` (Typer), `ui_app.py`
  (Gradio).

Pipeline stages, in order: Probe → Transcribe → Segment → Clip → Frame →
EmbedText → EmbedVision → Write. Each mutates a `PipelineContext`; the first
`is_ready() == False` stage ends the run for that video.

## Conventions

- `from __future__ import annotations` in every module; fully typed (`ty` gates).
- Ruff: `E,F,I,UP,B,SIM`, line length 100, target `py312`. `cli.py` is exempt
  from `B008` (Typer `Option(...)` defaults).
- Tests use a deterministic 10s fixture video (`conftest.py` `fixture_video`) and
  hash-based model fakes (`tests/_fakes.py`) — no real weights during `pytest`.
  Model loaders expose `_reset_cache_for_tests()`.
- Prefer editing in the style of the surrounding module (thin ffmpeg wrappers,
  small pure helpers, dataclasses for rows/results).

## Gotchas / non-obvious invariants

- **Blob columns** (`clip_bytes`, `keyframe_jpeg`) are Lance **Blob V1** — normal
  reads return descriptors, not bytes. Always read them via
  `store.read_segment_blob`.
- **Embedding-model guard**: ingest calls `store.assert_or_set_embedding_models`.
  Switching to a different model (even same dimension) against a populated store
  raises `EmbeddingModelMismatch` unless `--force`. Embedding dims are
  single-sourced in `schema.py` (`TEXT_EMBED_DIM` / `VISION_EMBED_DIM`).
- **Accurate seeking**: clips (`ClipStage`, `precise=True`) and keyframes are
  extracted with frame-accurate ffmpeg seeking so content aligns with the
  segment's `[start, end)` window. Don't reintroduce `-ss` before `-i` with
  `-c copy` for aligned output. Clips from older ingests are keyframe-snapped;
  fix with `--force` re-ingest.
- **Search scores**: hybrid (`text`) and `multi` fuse with RRF, then normalize to
  `(0, 1]` (`_normalize_fused_scores`) so scores are comparable to the cosine
  `visual` path. Raw per-source contributions stay in `SearchHit.components`.
- **`ensure_indexes`** decides status from `list_indices()` + `count_rows()`, not
  exception text; vector indexes are skipped below 256 rows (IVF training floor).
- **Gradio import** is at module top in `ui_app.py` (its introspection needs
  `gr.SelectData` resolvable); `cli.py` lazy-imports `ui_app.launch`, so the CLI
  doesn't pay the gradio import unless you launch the UI.
- The Search tab renders results as a **`gr.Dataframe` table** (not a gallery);
  `run_search` returns `(table_rows, raw_hits)` and a row click plays that
  segment's clip.

## Housekeeping

- Only commit/push when asked; branch off `main` first if needed.
- ffmpeg is required for most tests — they'll error clearly if it's missing.
