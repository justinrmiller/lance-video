# Development

## Environment

Requires Python **3.12+**, `ffmpeg` + `ffprobe` on `PATH`, and
[uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-groups     # install runtime + dev dependencies
```

## Quality gates

```bash
uv run pytest                     # unit tests (240 passed, 6 skipped, ~30s)
uv run ruff check src tests       # lint
uv run ruff format --check src tests scripts   # formatting
uv run ty check                   # static type check (Astral's `ty`)
```

Coverage:

```bash
uv run pytest --cov                       # terminal report
uv run pytest --cov --cov-report=html     # writes htmlcov/index.html
```

The 6 skipped tests are the opt-in real-model integration suite:

```bash
VL_INTEGRATION=1 uv run pytest tests/test_integration_real_models.py -v
```

### pre-commit

```bash
uv run pre-commit install         # wire the git hook (one-time)
uv run pre-commit run --all-files
```

Hooks: standard hygiene checks, `ruff-check --fix`, `ruff-format`, and `ty`
(scoped to `src/`). pytest is intentionally **not** a commit hook — run it
explicitly or wire it as a pre-push hook.

## Testing philosophy

- Every module has unit tests against a **deterministic 10s fixture video** (a
  color-bar + sine-wave clip built by `tests/fixtures/make_fixture.py`,
  exposed via the session-scoped `fixture_video` fixture in `conftest.py`).
- Model wrappers are exercised with **hash-based fakes** (`tests/_fakes.py`) — no
  real Whisper / e5 / SigLIP weights are downloaded during `pytest`. The opt-in
  integration suite is what validates the real model wiring.
- Pure logic (segmentation, RRF fusion, seek-arg construction, score
  normalization) is unit-tested directly, without touching ffmpeg or LanceDB.
- The process-cached model loaders expose `_reset_cache_for_tests()` so tests
  can start from a clean cache.

## Conventions

- `from __future__ import annotations` at the top of every module.
- Type everything; `ty` is the gate (mypy was dropped after `ty` reached parity).
- Ruff rule set: `E, F, I, UP, B, SIM`, line length 100, target `py312`. The CLI
  module is exempt from `B008` (Typer's `Option(...)` defaults are the canonical
  pattern).
- ffmpeg/ffprobe are located with `shutil.which` and invoked via `subprocess`
  with `-nostdin -loglevel error`; failures raise module-specific errors
  (`ProbeError`, `ClipError`, `FrameExtractError`).
- Blob columns are never read as plain columns — always via
  `store.read_segment_blob`.

## Extending the pipeline

### Add a stage
1. Implement a class with `name`, `is_ready`, and `run` (the `Stage` protocol in
   `stages.py`). Mutate `PipelineContext` in place; return a `StageResult`.
2. Guard `is_ready` on the inputs your stage needs *and* on `not ctx.skipped` so
   the idempotency skip still short-circuits.
3. Insert it into `DEFAULT_STAGES` at the right position. If it produces data
   `WriteStage` must persist, add the field to `WorkingSegment`, `SegmentRow`,
   the PyArrow schema in `schema.py`, and `WriteStage.is_ready`/`run`.

### Swap a model
- Change the default in `config.py` / the relevant `DEFAULT_*` constant, or pass
  `--text-embed-model` / `--vision-embed-model` / `--whisper-model`.
- If the embedding **dimension** changes, update `TEXT_EMBED_DIM` /
  `VISION_EMBED_DIM` in `schema.py` (the schema and `SegmentRow` validation read
  from there). A dimension change is a breaking store change — rebuild the DB.
- Switching to a different **same-dimension** model requires a full `--force`
  re-ingest; the embedding-model guard (`store.assert_or_set_embedding_models`)
  will otherwise reject the mismatch.

## Known limitations

- **Sequential per video.** A `ProcessPoolExecutor` is the natural next step but
  the embedder instances aren't easily pickleable — a follow-up.
- **Queries pay encoding cost each search** (e5 ~50 ms, SigLIP ~30 ms warm CPU) —
  negligible for humans, notable at high QPS.
- **`video_id` is path-based**, so moving a source file orphans its rows; a
  content-hash scheme is a planned upgrade.
- **Blob V1** is used because `lancedb`'s `create_table` still pins Lance file
  format 2.1; Blob V2 is a future upgrade.
