# CLI and UI reference

## CLI

The entry point is `video-lance` (Typer app in `cli.py`); run any command with
`uv run video-lance <command>`. Five commands: `ingest`, `search`, `info`,
`reindex`, `ui`.

### `ingest`

Index a directory (or a single video) into the store.

```bash
uv run video-lance ingest ./videos --segment-seconds 30 --db-path ./video-lance.db
```

| Flag | Default | Notes |
|---|---|---|
| `PATH` (arg) | — | Directory to walk, or a single video (must exist). |
| `--segment-seconds`, `-s` | `30.0` | Window length. |
| `--overlap-seconds`, `-o` | `0.0` | Adjacent-window overlap; must be `< segment_seconds`. |
| `--merge-short-tail / --no-merge-short-tail` | on | Fold a too-short final window into the previous one. |
| `--min-tail-seconds` | `5.0` | Threshold for the tail merge. |
| `--sentence-snap-tolerance` | `0.0` | Snap window edges to sentence ends within this many seconds (`0` = off). |
| `--frame-position` | `0.5` | Keyframe position within a window (`0`=start, `1`=end); clamped. |
| `--frame-jpeg-quality` | `85` | Keyframe JPEG quality. |
| `--frame-max-long-edge` | `512` | Downscale keyframe long edge to this. |
| `--workers` | `1` | Reserved; ingest is currently sequential. |
| `--device` | `auto` | `auto` / `cuda` / `mps` / `cpu`. |
| `--whisper-model` | `small.en` | faster-whisper checkpoint. |
| `--text-embed-model` | e5-instruct | Must match the store's model unless `--force`. |
| `--vision-embed-model` | siglip2-so400m | Must match the store's model unless `--force`. |
| `--db-path` | `./video-lance.db` | LanceDB location (local dir or `s3://…`). |
| `--include` | `*.mp4,*.mkv,*.mov,*.webm` | Comma-separated include globs. |
| `--exclude` | (none) | Comma-separated exclude globs. |
| `--force` | off | Re-ingest even if already indexed; also allows switching embedding models. |
| `--dry-run` | off | Walk + list matched files; no probe/write. |

Exit codes: `1` if nothing matched, `2` if any video failed (or on an embedding
model mismatch).

### `search`

```bash
uv run video-lance search "artificial intelligence" --mode text  --limit 5
uv run video-lance search "a person at a desk"       --mode visual
uv run video-lance search --image ./query.jpg        --mode visual
uv run video-lance search "computer chronicles"      --mode multi --visual-weight 0.4
```

| Flag | Default | Notes |
|---|---|---|
| `QUERY` (arg) | `""` | Text query; omit only when using `--image` in visual mode. |
| `--image` | — | Image file to use as a visual query. |
| `--mode` | `text` | `text` / `visual` / `multi`. |
| `--limit` | `10` | Max hits. |
| `--visual-weight` | `0.4` | `multi` only; weight on the SigLIP leg, in `[0, 1]`. |
| `--filter` | — | Raw SQL `WHERE` applied before fusion. |
| `--db-path` | `./video-lance.db` | Must exist. |
| `--device` | `auto` | Device for query encoding. |
| `--text-embed-model` / `--vision-embed-model` | store's, then default | Override the encoder used for the query. |

Model defaults come from what the DB was built with (the `_metadata` row), then
the built-in defaults. Output is a ranked list with score, path, time range, a
text snippet, and a `file://…#t=` deep link.

### `info`

Prints DB path, video/segment counts, the persisted embedding models, and every
index on the `segments` table.

### `reindex`

Drops and rebuilds the FTS + vector indexes (`ensure_indexes(replace=True)`).

### `ui`

Launches the Gradio app.

| Flag | Default |
|---|---|
| `--db-path` | `./video-lance.db` |
| `--host` | `127.0.0.1` |
| `--port` | `7860` |
| `--device` | `auto` |
| `--share` | off (ask Gradio for a temporary public URL) |

If the DB path doesn't exist, an empty store is created so the UI can launch cold
and you can ingest from the Ingest tab.

## UI

`build_context` opens the store and (lazily) loads the embedders; `build_app`
constructs the Blocks; `launch` starts the server. The footer ("Use via API /
Built with Gradio") is hidden via a `css` rule on `launch`.

The pure functions `run_search(ctx, query, mode, image, limit, sql_filter,
visual_weight)` and `play_clip(ctx, raw_hits, selected_index)` are plain Python
and unit-tested without Gradio (`tests/test_ui_app.py`).

### Search tab
- **Query** textbox, **Mode** radio (`text` / `visual` / `multi`), **Limit** and
  **Visual weight** sliders, an optional **Query image** upload, and an optional
  **SQL filter**.
- Results render as a **table** (`gr.Dataframe`, columns `# · score · path ·
  time · text`) above the video player. `run_search` returns `(table_rows,
  raw_hits)`; the raw hits are stashed in a `gr.State`.
- **Click a row** → `results_table.select` → `_on_select` resolves the row index
  against the stashed hits and `play_clip` streams that segment's `clip_bytes`
  into the `gr.Video` player above the table.

### Ingest tab
Runs the full pipeline from the browser. Point at a directory, set include /
exclude globs, **Discover** to preview matches, tune the segmentation / frame
sliders, then **Run ingest** — a streaming generator yields `(progress, log)`
after each video. **Cancel** terminates the generator between videos (the
in-flight video finishes first). Embedders + transcriber load lazily on first
run, so the Search tab never pays the Whisper load cost.

### Database tab
Stats markdown (counts, embedding models, indexes); a videos dataframe (click a
row to drill into its segments); a **Rebuild indexes** button; and a collapsed
"Danger zone" that deletes the selected video and its segments behind an "I
understand" checkbox.

> Note: the model identifiers used by the UI Ingest tab come from `ctx` (resolved
> from the store's `_metadata` or the defaults), so ingesting through the UI
> stays consistent with the store's embedding models.
