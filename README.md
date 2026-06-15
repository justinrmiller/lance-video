# video-lance

A Python pipeline that indexes a directory of videos into a LanceDB store with multi-modal embeddings, plus a Gradio UI for searching it.

```
                      ┌────────────────────────────────────────────────┐
                      │  uv run video-lance ingest ./videos            │
                      │                                                │
                      │  ffprobe → faster-whisper → segmenter →        │
                      │  ffmpeg clip/keyframe → e5-instruct (text) →   │
                      │  SigLIP 2 (vision) → LanceDB (Blob V2 for      │
                      │  clip_bytes + keyframe_jpeg)                   │
                      └─────────────────────┬──────────────────────────┘
                                            │
                                            ▼
                              ┌──────────────────────────┐
                              │  ./video-lance.db/       │
                              │   (or s3://...,          │
                              │    same Lance format)    │
                              └──────────┬───────────────┘
                                         │
                                         ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │  uv run video-lance ui   (Gradio Blocks)                        │
   │                                                                 │
   │  free-form text query → e5 encode_query                         │
   │  free-form text query → SigLIP 2 encode_text (cross-modal)      │
   │  image upload         → SigLIP 2 encode_image                   │
   │       ↓                                                         │
   │  LanceDB hybrid vector + FTS search                             │
   │       ↓                                                         │
   │  gallery of keyframe thumbnails (Blob V2 reads)                 │
   │  click a tile → inline <video> playback of clip_bytes           │
   └─────────────────────────────────────────────────────────────────┘
```

## Status

| Session | Scope | State |
|---|---|---|
| 1 | Scaffold, `models.py`, `config.py`, `segmenter.py` | ✅ done |
| 2 | `probe.py`, `transcribe.py`, `clipper.py`, `frames.py`, fixture video | ✅ done |
| 3 | `embed_text.py` (e5-instruct), `embed_vision.py` (SigLIP 2), device autodetect | ✅ done |
| 4 | PyArrow schemas, LanceDB store with Blob V2 | ✅ done |
| 5 | Stage protocol, pipeline orchestration, `video-lance ingest` CLI | ✅ done |
| 6 | Search (text / visual / multi), `info`, `reindex` | ✅ done |
| 7 | Gradio UI (`video-lance ui`) + HF Spaces shim | ✅ done |

The original PLAN.pdf called for a Streamlit UI; this repo briefly tried a Vite + React + serverless-TS-reader architecture, then collapsed to a single Gradio app — same end result (free-form search, gallery, inline playback), one language, no Node toolchain.

## Requirements

- **Python 3.11+**
- **ffmpeg + ffprobe** on `PATH` (`brew install ffmpeg` on macOS)
- **[uv](https://docs.astral.sh/uv/)** for environment management
- Optional: GPU (CUDA / MPS) — pipeline auto-detects, falls back to CPU

## Quickstart

```bash
git clone <this repo>
cd lance-video
uv sync
uv run pytest -v     # 207 tests, ~17s
```

### Ingest

```bash
uv run video-lance ingest ./videos \
    --segment-seconds 30 \
    --db-path ./video-lance.db
```

First run downloads Whisper + e5-instruct + SigLIP 2 weights to `~/.cache/huggingface` (~5–6 GB).

### Search from the CLI

```bash
uv run video-lance search "artificial intelligence" --mode text  --limit 5
uv run video-lance search "a person at a desk"      --mode visual --limit 5
uv run video-lance search --image ./query.jpg --mode visual
uv run video-lance search "computer chronicles"     --mode multi  --visual-weight 0.4
uv run video-lance info
uv run video-lance reindex
```

### Search from the web UI

```bash
uv run video-lance ui --db-path ./video-lance.db
# open http://127.0.0.1:7860
```

The UI is a Gradio Blocks app with three tabs:

**Search**
- **Free-form text query** — no menu, no pre-computed vectors. Each query is encoded server-side on click.
- **Mode switcher** — `text` (e5 + FTS hybrid), `visual` (SigLIP 2 cross-modal), `multi` (RRF blend of both).
- **Image upload** — drop in any image; SigLIP 2 encodes it and searches against `visual_embedding`.
- **SQL filter** — pass a `WHERE` expression like `duration_s > 60` straight to LanceDB.
- **Gallery of keyframe thumbnails** — pulled from the `keyframe_jpeg` Blob V2 column on every search.
- **Inline clip playback** — click a tile; the segment's `clip_bytes` Blob V2 column is read out, written to a tempfile, and fed to `<video>` with autoplay.

**Ingest** — run the full pipeline from the browser.
- Point at a videos directory, set include / exclude globs, hit **Discover** to preview the file list.
- All `SegmentationConfig` and `FrameSamplingConfig` knobs from the CLI, plus `--force`, are exposed as sliders / checkboxes.
- **Run ingest** kicks off a streaming generator that yields after each video: a live progress bar and a growing log keep you informed. **Cancel** terminates the generator (in-flight video finishes before the cancellation takes effect).
- Embedders + transcriber are loaded lazily on first run — the Search tab doesn't pay the Whisper load cost.

**Database** — administer the LanceDB store.
- Top: stats Markdown (row counts, persisted embedding model identifiers, every index on the segments table).
- Middle: videos dataframe; click a row to populate the segments dataframe below it.
- Danger zone (collapsed accordion): delete the selected video + its segments, gated by an "I understand…" checkbox.
- **Rebuild indexes** button = `video-lance reindex` (drops + creates FTS + IVF on both embedding columns).

### Deploy to Hugging Face Spaces

`ui/app.py` is the Spaces entry point. It reads the LanceDB store from `VL_DB_PATH` (env var, default `./video-lance.db`). For a real deploy:

1. Build the LanceDB store locally with `video-lance ingest`.
2. Upload it to S3 / R2 / your bucket of choice (Lance natively supports `s3://...` URIs).
3. Create a Hugging Face Space with the Gradio SDK and point it at this repo.
4. Set Space secrets:
   - `VL_DB_PATH=s3://your-bucket/video-lance.db`
   - `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` (or the IAM-role equivalents).
5. Push. The Space's `app.py` is the shim that imports `video_lance.ui_app` and launches it.

The data is still S3-backed; only the *runtime* moves from "Lambda function in your account" to "Spaces container managed by HF." Trade-off discussed in earlier session history if you want the receipts.

## Repository layout

```
video-lance/
├── pyproject.toml
├── scripts/demo.sh                # one-shot ingest + search demo (CLI only)
├── ui/app.py                      # Hugging Face Spaces entry shim
├── src/video_lance/
│   ├── config.py                  # SegmentationConfig, FrameSamplingConfig, Config
│   ├── models.py                  # TranscriptWord, Transcript, VideoMeta
│   ├── device.py                  # autodetect_device / resolve_device
│   ├── segmenter.py               # compute_segments — pure, no I/O
│   ├── probe.py                   # ffprobe wrapper -> VideoMeta
│   ├── clipper.py                 # ffmpeg clip extraction -> mp4 bytes
│   ├── frames.py                  # ffmpeg single-frame extraction -> (jpeg, PIL.Image)
│   ├── transcribe.py              # faster-whisper wrapper + map_text_to_window
│   ├── embed_text.py              # e5-instruct (1024-d, L2-normalized)
│   ├── embed_vision.py            # SigLIP 2 (1152-d, L2-normalized, cross-modal)
│   ├── discovery.py               # walk dir, filter by include/exclude globs
│   ├── schema.py                  # PyArrow schemas + dim constants + Blob V2 tags
│   ├── store.py                   # LanceDB connect / upsert / blob reads
│   ├── stages.py                  # Stage protocol + 8 concrete stages
│   ├── pipeline.py                # process_video / process_directory
│   ├── search.py                  # text / visual / multi + ensure_indexes + db_info
│   ├── rerank.py                  # cross-encoder rerank stub (raises)
│   ├── ui_app.py                  # Gradio Blocks app + run_search / play_clip
│   └── cli.py                     # typer: ingest / search / info / reindex / ui
└── tests/                         # 207 tests (+ 6 opt-in integration tests)
    ├── _fakes.py                  # shared fake Whisper / e5 / SigLIP 2
    ├── conftest.py                # session-scoped fixture_video
    └── fixtures/make_fixture.py   # deterministic 10s color-bar + sine-wave video
```

## How the UI actually works

`run_search(ctx, query, mode, image, limit, sql_filter, visual_weight)` and `play_clip(ctx, raw_hits, selected_index)` are plain Python functions on `src/video_lance/ui_app.py`. They're independently unit-tested (`tests/test_ui_app.py`). Gradio is just the I/O layer:

- Submit / `Search` button → `run_search` → returns `(gallery_rows, raw_hits_state)`.
- `gallery.select` → `play_clip` reads the segment's `clip_bytes` blob and returns a tempfile path to the `gr.Video` component.

The Gradio import is at module top because Gradio's introspection calls `typing.get_type_hints()` on event handlers, and `gr.SelectData` annotations can't be resolved if `gr` is only bound inside a function. Once is enough; the CLI lazy-imports `ui_app.launch`, so the ~0.8 s gradio import only fires when you actually launch the UI (or run the UI tests).

## Testing

```bash
uv run pytest -v                            # 226 passed
uv run ruff check src tests                 # lint
uv run ty check                             # static type check (Astral's `ty`)

# Coverage (branch + missing lines; baseline ~89% across the package).
uv run pytest --cov                         # terminal report
uv run pytest --cov --cov-report=html       # writes htmlcov/index.html

# Opt-in: hits the real model weights (~6 GB downloads on first run).
VL_INTEGRATION=1 uv run pytest tests/test_integration_real_models.py -v
```

Coverage is configured in `pyproject.toml` (`[tool.coverage.run]` / `[tool.coverage.report]`). `source = ["src/video_lance"]`, branch coverage on, and the usual sentinel lines (`if TYPE_CHECKING:`, `raise NotImplementedError`, `if __name__ == .__main__.:`, `pragma: no cover`) are excluded. The Gradio `build_app` / `launch` paths and the optional CUDA / MPS branches in `device.py` are the main uncovered regions — those need real hardware or a live server to exercise.

### pre-commit

```bash
uv run pre-commit install            # one-time: wires the git pre-commit hook
uv run pre-commit run --all-files    # run every hook against the whole tree
```

`.pre-commit-config.yaml` ships with:

- the standard hygiene hooks (`trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-toml`, `check-added-large-files`, `check-merge-conflict`, `mixed-line-ending`),
- `ruff-check --fix` (auto-fix on commit),
- `ty` (Astral's type checker, scoped to `src/` via `[tool.ty.src]`).

pytest is intentionally **not** a hook — at ~20 s it'd make `git commit` painful. Run it explicitly, or wire it as a pre-push hook if you want belt-and-braces.

Every Python module has unit tests against a deterministic 10 s color-bar fixture video. The model wrappers are tested with hash-based fakes (no real Whisper / e5 / SigLIP 2 weights downloaded during `pytest`). The opt-in integration suite loads the real models and is what would have caught the `sentencepiece` and `transformers 5.x ModelOutput` bugs earlier.

## Known limitations

- **Pipeline is sequential per video.** PLAN §6 called for a `ProcessPoolExecutor`; the embedder instances aren't easily pickleable, so adding it sensibly is a follow-up.
- **Free-form queries pay encoding cost on every search.** e5 ~50 ms, SigLIP 2 ~30 ms on warm CPU. Imperceptible to humans but worth noting if you go very high QPS.
- **SigLIP 2 is multilingual** (a big improvement over SigLIP 1, which was English-centric). Non-English `visual` / `multi` queries work reasonably; for transcript-over-transcript text search, e5-instruct remains the better encoder, which is why the `text` mode keeps using it.
- **No live ingest from the UI.** The UI only reads. To re-ingest with new settings, drop to the CLI.
