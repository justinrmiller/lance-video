"""Gradio web UI for video-lance.

Run via the CLI:

    uv run video-lance ui --db-path ./video-lance.db

…or directly:

    uv run python -m video_lance.ui_app

For Hugging Face Spaces, point the Space's app file at `ui/app.py` (the
top-level shim), which delegates here.

Design notes
------------
The pure-function entrypoints (`run_search`, `play_clip`, `build_context`)
are exposed at module level so they're unit-testable *without* spinning up
Gradio. `build_app` and `launch` import Gradio lazily — they're only called
from the CLI and from the HF Spaces shim.
"""

from __future__ import annotations

import io
import logging
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gradio as gr
from PIL import Image

from video_lance import discovery, pipeline, store
from video_lance import search as search_mod
from video_lance.config import (
    DEFAULT_INCLUDE,
    Config,
    FrameSamplingConfig,
    SegmentationConfig,
)
from video_lance.device import resolve_device
from video_lance.embed_text import (
    DEFAULT_TEXT_MODEL,
    E5Embedder,
    get_text_embedder,
)
from video_lance.embed_vision import (
    DEFAULT_VISION_MODEL,
    SigLIPEmbedder,
    get_vision_embedder,
)
from video_lance.transcribe import WhisperTranscriber, get_transcriber

logger = logging.getLogger(__name__)

ALLOWED_MODES = ("text", "visual", "multi")
DEFAULT_WHISPER_MODEL = "small.en"


# -- context ------------------------------------------------------------------


@dataclass
class AppContext:
    """The mutable state the Gradio callbacks read from. Built once per
    process. Test code substitutes embedders directly.

    `device` is the *resolved* device (`cpu` / `cuda` / `mps`), not `"auto"`,
    so callers lazy-loading a Whisper transcriber for the Ingest tab can
    pass it through directly.
    """

    db_path: Path
    tables: store.StoreTables
    text_embedder: E5Embedder
    vision_embedder: SigLIPEmbedder
    device: str = "cpu"
    whisper_model: str = DEFAULT_WHISPER_MODEL

    def get_transcriber(self) -> WhisperTranscriber:
        """Lazy: only load Whisper weights when the Ingest tab actually runs.
        `get_transcriber` is process-cached so repeat calls are free."""
        return get_transcriber(self.whisper_model, device=self.device)


def build_context(
    db_path: Path,
    *,
    device: str = "auto",
    text_model: str | None = None,
    vision_model: str | None = None,
    whisper_model: str = DEFAULT_WHISPER_MODEL,
) -> AppContext:
    """Open the LanceDB store and load the embedders.

    Model identifiers fall back to whatever the DB was built with (read from
    the persisted metadata table), then to the project defaults — same
    precedence the `search` CLI uses.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(
            f"no LanceDB store at {db_path}; run `video-lance ingest` first."
        )

    resolved = resolve_device(device)

    db = store.connect(db_path)
    tables = store.ensure_tables(db)
    stored_text, stored_vision = store.get_embedding_models(tables)

    text_emb = get_text_embedder(
        text_model or stored_text or DEFAULT_TEXT_MODEL,
        device=resolved,
    )
    vision_emb = get_vision_embedder(
        vision_model or stored_vision or DEFAULT_VISION_MODEL,
        device=resolved,
    )
    return AppContext(
        db_path=db_path,
        tables=tables,
        text_embedder=text_emb,
        vision_embedder=vision_emb,
        device=resolved,
        whisper_model=whisper_model,
    )


# -- search -------------------------------------------------------------------


def _hit_to_state(hit: search_mod.SearchHit) -> dict[str, Any]:
    """Pickle-clean dict for the gallery's hidden state row."""
    return {
        "segment_id": hit.segment_id,
        "video_id": hit.video_id,
        "idx": hit.idx,
        "start_s": hit.start_s,
        "end_s": hit.end_s,
        "text": hit.text,
        "score": hit.score,
        "source_path": hit.source_path,
        "relative_path": hit.relative_path,
        "components": dict(hit.components),
    }


def _placeholder_thumb() -> Image.Image:
    return Image.new("RGB", (160, 120), (60, 60, 60))


def _load_thumb(tables: store.StoreTables, segment_id: str) -> Image.Image:
    try:
        jpeg = store.read_segment_blob(tables, segment_id, "keyframe_jpeg")
    except KeyError:
        logger.warning("no keyframe_jpeg blob for %s", segment_id)
        return _placeholder_thumb()
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to read keyframe for %s: %s", segment_id, exc)
        return _placeholder_thumb()

    try:
        return Image.open(io.BytesIO(jpeg)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        logger.warning("invalid JPEG for %s: %s", segment_id, exc)
        return _placeholder_thumb()


def _caption(rank: int, hit: search_mod.SearchHit) -> str:
    path = hit.relative_path or hit.video_id
    snippet = (hit.text or "").strip().replace("\n", " ")
    if len(snippet) > 90:
        snippet = snippet[:87] + "..."
    parts = [
        f"{rank}. [{hit.score:.3f}] {path}",
        f"   {hit.time_range()}",
    ]
    if snippet:
        parts.append(f'   "{snippet}"')
    return "\n".join(parts)


def run_search(
    ctx: AppContext,
    query: str,
    mode: str,
    image: Image.Image | None,
    limit: int,
    sql_filter: str,
    visual_weight: float,
) -> tuple[list[tuple[Image.Image, str]], list[dict[str, Any]]]:
    """Execute a search, return `(gallery_rows, raw_hits)`.

    `gallery_rows` is a list of `(PIL image, caption)` tuples for `gr.Gallery`.
    `raw_hits` is a parallel list of dicts the UI stashes in a `gr.State` so
    later events (click → play clip) can resolve the chosen segment back to
    its LanceDB row.
    """
    mode_clean = (mode or "").strip().lower()
    if mode_clean not in ALLOWED_MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {ALLOWED_MODES}")

    sql = (sql_filter or "").strip() or None
    n = max(1, int(limit))

    if mode_clean == "text":
        if not query.strip():
            raise ValueError("text mode requires a query string")
        hits = search_mod.search_text(
            ctx.tables, ctx.text_embedder, query, limit=n, sql_filter=sql
        )
    elif mode_clean == "visual":
        if image is not None:
            hits = search_mod.search_visual(
                ctx.tables,
                ctx.vision_embedder,
                image=image,
                limit=n,
                sql_filter=sql,
            )
        elif query.strip():
            hits = search_mod.search_visual(
                ctx.tables,
                ctx.vision_embedder,
                query=query,
                limit=n,
                sql_filter=sql,
            )
        else:
            raise ValueError("visual mode requires a query string or an image")
    else:  # multi
        if not query.strip():
            raise ValueError("multi mode requires a query string")
        hits = search_mod.search_multi(
            ctx.tables,
            ctx.text_embedder,
            ctx.vision_embedder,
            query,
            limit=n,
            visual_weight=float(visual_weight),
            sql_filter=sql,
        )

    gallery: list[tuple[Image.Image, str]] = []
    raw: list[dict[str, Any]] = []
    for rank, hit in enumerate(hits, start=1):
        gallery.append((_load_thumb(ctx.tables, hit.segment_id), _caption(rank, hit)))
        raw.append(_hit_to_state(hit))
    return gallery, raw


# -- clip playback ------------------------------------------------------------


def play_clip(
    ctx: AppContext,
    raw_hits: list[dict[str, Any]],
    selected_index: int | None,
) -> str | None:
    """Resolve a gallery click to a playable MP4 path.

    Reads the segment's `clip_bytes` Blob V2 column out of LanceDB, writes
    it to a tempfile (Gradio's `gr.Video` wants a path), and returns the
    path. Returns `None` if the index is out of range or the segment has no
    clip stored.

    Tempfiles are intentionally not cleaned up — Gradio re-uses the same
    path while the player is showing the clip, and the OS will reclaim
    `/tmp` eventually.
    """
    if selected_index is None or not raw_hits:
        return None
    if selected_index < 0 or selected_index >= len(raw_hits):
        return None
    hit = raw_hits[selected_index]
    try:
        clip = store.read_segment_blob(ctx.tables, hit["segment_id"], "clip_bytes")
    except KeyError:
        logger.warning("no clip_bytes for %s", hit["segment_id"])
        return None

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(clip)
        return tmp.name


# -- database view ------------------------------------------------------------

VIDEOS_COLUMNS = [
    "video_id",
    "relative_path",
    "duration_s",
    "fps",
    "width",
    "height",
    "size_mb",
    "segment_seconds",
    "segments",
]
SEGMENTS_COLUMNS = ["idx", "start_s", "end_s", "text"]


def list_videos(tables: store.StoreTables) -> list[dict[str, Any]]:
    """All rows from the `videos` table, augmented with each video's segment
    count. Used by the Database tab's top dataframe."""
    rows = tables.videos.search().limit(1_000_000).to_arrow().to_pylist()
    out: list[dict[str, Any]] = []
    for r in rows:
        segs = store.get_segments_for_video(tables, str(r["video_id"]))
        size_bytes = int(r.get("size_bytes") or 0)
        out.append(
            {
                "video_id": str(r["video_id"]),
                "relative_path": str(r.get("relative_path") or ""),
                "duration_s": float(r.get("duration_s") or 0.0),
                "fps": float(r.get("fps") or 0.0),
                "width": int(r.get("width") or 0),
                "height": int(r.get("height") or 0),
                "size_mb": round(size_bytes / (1024 * 1024), 2),
                "segment_seconds": float(r.get("segment_seconds") or 0.0),
                "segments": len(segs),
            }
        )
    return out


def list_segments_for_video(
    tables: store.StoreTables,
    video_id: str,
) -> list[dict[str, Any]]:
    """Lightweight segment rows for the Database tab's drill-down dataframe.
    Embeddings and blob descriptors are intentionally dropped."""
    rows = store.get_segments_for_video(tables, video_id)
    return [
        {
            "idx": int(r["idx"]),
            "start_s": float(r["start_s"]),
            "end_s": float(r["end_s"]),
            "text": (str(r.get("text") or "")[:160]),
        }
        for r in sorted(rows, key=lambda r: int(r["idx"]))
    ]


def db_stats_markdown(tables: store.StoreTables, db_path: Path | None = None) -> str:
    """One-shot Markdown summary for the Database tab header."""
    info = search_mod.db_info(tables, db_path=db_path)
    indices = "\n".join(f"  - {n}" for n in info.segment_indexes) or "  _(none built)_"
    return (
        f"**db**: `{info.db_path}`\n\n"
        f"- videos: **{info.videos}**\n"
        f"- segments: **{info.segments}**\n"
        f"- text embed model: `{info.text_embed_model or '(not set)'}`\n"
        f"- vision embed model: `{info.vision_embed_model or '(not set)'}`\n\n"
        f"**segment indexes**\n{indices}"
    )


def delete_video_action(
    tables: store.StoreTables,
    video_id: str,
    *,
    confirm: bool,
) -> str:
    """Delete a video + its segments. The Database tab gates this behind a
    confirmation checkbox; we re-check here so callers (including tests)
    can't accidentally call the destructive path."""
    if not video_id:
        return "no video selected"
    if not confirm:
        return "confirmation required — tick the 'I understand' box and try again"
    existing = store.get_video(tables, video_id)
    if existing is None:
        return f"no row with video_id={video_id!r}"
    store.delete_video(tables, video_id)
    return f"deleted {video_id} ({existing.get('relative_path') or ''})"


def rebuild_indexes_action(tables: store.StoreTables) -> str:
    """Rebuild FTS + vector indexes. Returns a short status string."""
    status = search_mod.ensure_indexes(tables, replace=True)
    return (
        "rebuilt:\n"
        f"  - fts_text:           {status.fts_text}\n"
        f"  - vec_text_embedding: {status.vec_text}\n"
        f"  - vec_visual_embed:   {status.vec_visual}"
    )


# -- ingest view --------------------------------------------------------------

DISCOVER_COLUMNS = ["name", "relative_path", "size_mb"]


def discover_for_table(
    root: Path,
    *,
    include: tuple[str, ...] = DEFAULT_INCLUDE,
    exclude: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """Walk `root` and return a row per matching file for the Ingest tab's
    preview dataframe. Pure — no DB / embedder access.

    Empty / whitespace-only `root` returns `[]` (Python's `Path("").exists()`
    is `True` and resolves to the current working directory, which we
    deliberately don't want to walk by accident from the UI's blank-textbox
    state)."""
    s = str(root).strip()
    if not s:
        return []
    base = Path(s)
    if not base.exists():
        return []
    base = base.resolve()
    out: list[dict[str, Any]] = []
    for path in discovery.walk(base, include=include, exclude=exclude):
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        try:
            rel = str(path.relative_to(base))
        except ValueError:
            rel = path.name
        out.append(
            {
                "name": path.name,
                "relative_path": rel,
                "size_mb": round(size / (1024 * 1024), 2),
            }
        )
    return out


def _build_ingest_config(
    *,
    segment_seconds: float,
    overlap_seconds: float,
    merge_short_tail: bool,
    sentence_snap_tolerance: float,
    frame_position: float,
    frame_jpeg_quality: int,
    frame_max_long_edge: int,
    device: str,
    whisper_model: str,
    text_model: str,
    vision_model: str,
) -> Config:
    return Config(
        segmentation=SegmentationConfig(
            segment_seconds=segment_seconds,
            overlap_seconds=overlap_seconds,
            merge_short_tail=merge_short_tail,
            sentence_snap_tolerance_seconds=sentence_snap_tolerance,
        ),
        frames=FrameSamplingConfig(
            position=frame_position,
            jpeg_quality=frame_jpeg_quality,
            max_long_edge=frame_max_long_edge,
        ),
        whisper_model=whisper_model,
        text_embed_model=text_model,
        vision_embed_model=vision_model,
        device=device,
    )


def run_ingest_streaming(
    ctx: AppContext,
    root: Path,
    *,
    segment_seconds: float = 30.0,
    overlap_seconds: float = 0.0,
    merge_short_tail: bool = True,
    sentence_snap_tolerance: float = 0.0,
    frame_position: float = 0.5,
    frame_jpeg_quality: int = 85,
    frame_max_long_edge: int = 512,
    force: bool = False,
) -> Iterator[tuple[float, str]]:
    """Yield `(progress_fraction, log_text)` after each video.

    Gradio wires this as a streaming generator: every yield updates the UI's
    progress slider and log textbox. The Cancel button is registered with
    `cancels=[run_event]`, so Gradio terminates the generator between yields
    — no custom cancellation Event needed.

    All embedder model identifiers are taken from `ctx` so re-ingesting
    against a DB that was built with different models is rejected by the
    persistent `_metadata` row, not silently rebuilt. The `whisper_model`
    can differ across runs (it's not stored).
    """
    s = str(root).strip()
    if not s:
        yield 1.0, "no path provided"
        return
    root = Path(s)
    if not root.exists():
        yield 1.0, f"path does not exist: {root}"
        return

    cfg = _build_ingest_config(
        segment_seconds=segment_seconds,
        overlap_seconds=overlap_seconds,
        merge_short_tail=merge_short_tail,
        sentence_snap_tolerance=sentence_snap_tolerance,
        frame_position=frame_position,
        frame_jpeg_quality=frame_jpeg_quality,
        frame_max_long_edge=frame_max_long_edge,
        device=ctx.device,
        whisper_model=ctx.whisper_model,
        text_model=ctx.text_embedder.model_name,
        vision_model=ctx.vision_embedder.model_name,
    )

    paths = discovery.walk(root, include=cfg.include, exclude=cfg.exclude)
    if not paths:
        yield 1.0, f"no videos matched under {root}"
        return

    transcriber = ctx.get_transcriber()
    store.set_embedding_models(
        ctx.tables,
        text_embed_model=cfg.text_embed_model,
        vision_embed_model=cfg.vision_embed_model,
    )

    log: list[str] = [f"discovered {len(paths)} video(s) under {root}"]
    n_ok = 0
    n_skip = 0
    n_fail = 0
    n_segments = 0
    yield 0.0, "\n".join(log)

    for i, path in enumerate(paths):
        result = pipeline.process_video(
            path,
            root,
            cfg,
            ctx.tables,
            transcriber=transcriber,
            text_embedder=ctx.text_embedder,
            vision_embedder=ctx.vision_embedder,
            force=force,
        )
        if result.ok and not result.skipped:
            n_ok += 1
            n_segments += result.segments_written
            log.append(f"  ok      {path.name}  ({result.segments_written} segments)")
        elif result.skipped:
            n_skip += 1
            log.append(f"  skip    {path.name}  ({result.skip_reason or 'already indexed'})")
        else:
            n_fail += 1
            log.append(f"  FAILED  {path.name}  {result.error}")
        yield (i + 1) / len(paths), "\n".join(log)

    if n_ok > 0:
        idx_status = search_mod.ensure_indexes(ctx.tables, replace=False)
        log.append(
            f"  indexes: fts_text={idx_status.fts_text} "
            f"vec_text={idx_status.vec_text} vec_visual={idx_status.vec_visual}"
        )

    log.append(
        f"done — succeeded={n_ok} skipped={n_skip} "
        f"failed={n_fail} segments_written={n_segments}"
    )
    yield 1.0, "\n".join(log)


# -- Gradio app (lazy import) -------------------------------------------------


def build_app(ctx: AppContext) -> Any:
    """Construct the Gradio Blocks.

    `gradio` is imported at module top because Gradio's own introspection
    calls `typing.get_type_hints()` on event handlers — if `gr` lives only
    in a function-local scope, `gr.SelectData` can't be resolved from the
    handler's `__globals__` and Gradio emits a spurious warning. Module-top
    import keeps the type annotations resolvable.
    """
    n_segments = int(ctx.tables.segments.count_rows())
    n_videos = int(ctx.tables.videos.count_rows())

    def _on_search(
        query: str,
        mode: str,
        image: Image.Image | None,
        limit: int,
        sql_filter: str,
        visual_weight: float,
    ) -> tuple[list[tuple[Image.Image, str]], list[dict[str, Any]]]:
        try:
            return run_search(ctx, query, mode, image, limit, sql_filter, visual_weight)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc

    def _on_select(
        raw_hits: list[dict[str, Any]] | None,
        evt: gr.SelectData,
    ) -> str | None:
        idx: Any = evt.index if evt is not None else None
        if isinstance(idx, (list, tuple)) and idx:
            idx = idx[0]
        if not isinstance(idx, int):
            return None
        return play_clip(ctx, raw_hits or [], idx)

    # -- Database tab callbacks --
    def _on_video_select(
        videos_data: Any,
        evt: gr.SelectData,
    ) -> tuple[str, list[list[Any]]]:
        """Resolve a click on the videos dataframe to (selected video_id,
        segment rows for that video)."""
        idx: Any = evt.index if evt is not None else None
        if isinstance(idx, (list, tuple)) and idx:
            idx = idx[0]
        if not isinstance(idx, int):
            return "", []
        rows = list_videos(ctx.tables)
        if idx < 0 or idx >= len(rows):
            return "", []
        vid = rows[idx]["video_id"]
        segs = list_segments_for_video(ctx.tables, vid)
        return vid, [[r[c] for c in SEGMENTS_COLUMNS] for r in segs]

    def _on_refresh_db() -> tuple[str, list[list[Any]], list[list[Any]], str]:
        rows = list_videos(ctx.tables)
        videos_df = [[r[c] for c in VIDEOS_COLUMNS] for r in rows]
        return db_stats_markdown(ctx.tables, ctx.db_path), videos_df, [], ""

    def _on_delete_video(video_id: str, confirm: bool) -> tuple[str, list[list[Any]]]:
        status = delete_video_action(ctx.tables, video_id, confirm=confirm)
        rows = list_videos(ctx.tables)
        videos_df = [[r[c] for c in VIDEOS_COLUMNS] for r in rows]
        return status, videos_df

    def _on_rebuild_indexes() -> str:
        return rebuild_indexes_action(ctx.tables)

    # -- Ingest tab callbacks --
    def _on_discover(root: str, include: str, exclude: str) -> list[list[Any]]:
        inc = tuple(p.strip() for p in include.split(",") if p.strip()) or DEFAULT_INCLUDE
        exc = tuple(p.strip() for p in exclude.split(",") if p.strip())
        if not root:
            return []
        rows = discover_for_table(Path(root), include=inc, exclude=exc)
        return [[r[c] for c in DISCOVER_COLUMNS] for r in rows]

    def _on_ingest(
        root: str,
        segment_seconds: float,
        overlap_seconds: float,
        merge_short_tail: bool,
        sentence_snap_tolerance: float,
        frame_position: float,
        frame_jpeg_quality: int,
        frame_max_long_edge: int,
        force: bool,
    ) -> Iterator[tuple[float, str]]:
        if not root:
            yield 1.0, "no path provided"
            return
        yield from run_ingest_streaming(
            ctx,
            Path(root),
            segment_seconds=segment_seconds,
            overlap_seconds=overlap_seconds,
            merge_short_tail=merge_short_tail,
            sentence_snap_tolerance=sentence_snap_tolerance,
            frame_position=frame_position,
            frame_jpeg_quality=frame_jpeg_quality,
            frame_max_long_edge=frame_max_long_edge,
            force=force,
        )

    # Theme moved from Blocks() to launch() in Gradio 6.x.
    with gr.Blocks(title="video-lance") as demo:
        gr.Markdown(
            f"# 🎞️ video-lance\n"
            f"_{n_videos} video(s), {n_segments} segments at app start — "
            f"LanceDB search at runtime, ingest available via the Ingest tab._"
        )

        with gr.Tabs():
            # =========================== SEARCH ===========================
            with gr.Tab("Search"):
                with gr.Row():
                    with gr.Column(scale=2):
                        query_in = gr.Textbox(
                            label="Query",
                            placeholder="e.g. artificial intelligence",
                            lines=1,
                        )
                        mode_in = gr.Radio(
                            list(ALLOWED_MODES),
                            value="text",
                            label="Mode",
                            info=(
                                "text = hybrid e5+FTS · visual = SigLIP cross-modal · "
                                "multi = RRF blend"
                            ),
                        )
                        with gr.Row():
                            limit_in = gr.Slider(1, 30, value=10, step=1, label="Limit")
                            visual_weight_in = gr.Slider(
                                0.0, 1.0, value=0.4, step=0.05, label="Visual weight (multi)"
                            )
                        image_in = gr.Image(
                            label="Query image (visual mode, optional)",
                            type="pil",
                            height=200,
                        )
                        filter_in = gr.Textbox(
                            label="SQL filter (optional)",
                            placeholder="duration_s > 60",
                        )
                        search_btn = gr.Button("Search", variant="primary")
                    with gr.Column(scale=3):
                        gallery = gr.Gallery(
                            label="Results · click a tile to play the clip",
                            columns=3,
                            height=520,
                            object_fit="cover",
                            show_label=True,
                        )
                        video = gr.Video(
                            label="Selected clip",
                            height=320,
                            autoplay=True,
                        )

                raw_hits_state = gr.State([])

                search_btn.click(
                    fn=_on_search,
                    inputs=[
                        query_in,
                        mode_in,
                        image_in,
                        limit_in,
                        filter_in,
                        visual_weight_in,
                    ],
                    outputs=[gallery, raw_hits_state],
                )
                query_in.submit(
                    fn=_on_search,
                    inputs=[
                        query_in,
                        mode_in,
                        image_in,
                        limit_in,
                        filter_in,
                        visual_weight_in,
                    ],
                    outputs=[gallery, raw_hits_state],
                )
                gallery.select(
                    fn=_on_select,
                    inputs=[raw_hits_state],
                    outputs=[video],
                )

            # =========================== INGEST ===========================
            with gr.Tab("Ingest"):
                gr.Markdown(
                    "Walk a directory, run the full pipeline (transcribe → segment → "
                    "embed → write to LanceDB). Cancel mid-run if needed; in-flight "
                    "videos finish before the cancellation takes effect."
                )
                ingest_root = gr.Textbox(
                    label="Videos directory",
                    placeholder="./videos",
                    lines=1,
                )
                with gr.Row():
                    include_in = gr.Textbox(
                        label="Include globs",
                        value=",".join(DEFAULT_INCLUDE),
                    )
                    exclude_in = gr.Textbox(
                        label="Exclude globs",
                        value="",
                    )
                discover_btn = gr.Button("Discover")
                discover_table = gr.Dataframe(
                    headers=DISCOVER_COLUMNS,
                    label="Discovered files (preview)",
                    interactive=False,
                    wrap=True,
                )

                with gr.Accordion("Segmentation", open=True):
                    with gr.Row():
                        segment_seconds_in = gr.Slider(
                            1.0, 120.0, value=30.0, step=1.0, label="segment_seconds"
                        )
                        overlap_seconds_in = gr.Slider(
                            0.0, 30.0, value=0.0, step=0.5, label="overlap_seconds"
                        )
                    with gr.Row():
                        merge_short_tail_in = gr.Checkbox(
                            value=True, label="merge_short_tail"
                        )
                        sentence_snap_in = gr.Slider(
                            0.0, 5.0, value=0.0, step=0.1,
                            label="sentence_snap_tolerance_s",
                        )

                with gr.Accordion("Frame sampling", open=False), gr.Row():
                    frame_position_in = gr.Slider(
                        0.0, 1.0, value=0.5, step=0.05, label="position"
                    )
                    jpeg_quality_in = gr.Slider(
                        1, 100, value=85, step=1, label="jpeg_quality"
                    )
                    max_edge_in = gr.Slider(
                        64, 1024, value=512, step=32, label="max_long_edge"
                    )

                force_in = gr.Checkbox(value=False, label="--force (re-ingest if already indexed)")

                with gr.Row():
                    run_btn = gr.Button("Run ingest", variant="primary")
                    cancel_btn = gr.Button("Cancel", variant="stop")

                progress_out = gr.Slider(
                    0.0, 1.0, value=0.0, step=0.001,
                    label="Progress",
                    interactive=False,
                )
                log_out = gr.Textbox(
                    label="Log",
                    lines=14,
                    interactive=False,
                )

                discover_btn.click(
                    fn=_on_discover,
                    inputs=[ingest_root, include_in, exclude_in],
                    outputs=[discover_table],
                )

                ingest_event = run_btn.click(
                    fn=_on_ingest,
                    inputs=[
                        ingest_root,
                        segment_seconds_in,
                        overlap_seconds_in,
                        merge_short_tail_in,
                        sentence_snap_in,
                        frame_position_in,
                        jpeg_quality_in,
                        max_edge_in,
                        force_in,
                    ],
                    outputs=[progress_out, log_out],
                )
                cancel_btn.click(fn=None, inputs=None, outputs=None, cancels=[ingest_event])

            # ========================== DATABASE ==========================
            with gr.Tab("Database"):
                with gr.Row():
                    refresh_btn = gr.Button("Refresh")
                    reindex_btn = gr.Button("Rebuild indexes")
                stats_md = gr.Markdown(db_stats_markdown(ctx.tables, ctx.db_path))
                reindex_status = gr.Textbox(label="Reindex status", lines=4, interactive=False)

                initial_videos_rows = list_videos(ctx.tables)
                videos_table = gr.Dataframe(
                    value=[[r[c] for c in VIDEOS_COLUMNS] for r in initial_videos_rows],
                    headers=VIDEOS_COLUMNS,
                    label="Videos · click a row to drill in",
                    interactive=False,
                    wrap=True,
                )
                selected_vid = gr.Textbox(label="Selected video_id", interactive=False)
                segments_table = gr.Dataframe(
                    headers=SEGMENTS_COLUMNS,
                    label="Segments for selected video",
                    interactive=False,
                    wrap=True,
                )

                with gr.Accordion("Danger zone", open=False):
                    confirm_delete = gr.Checkbox(
                        value=False,
                        label=(
                            "I understand this will permanently delete the "
                            "selected video and its segments"
                        ),
                    )
                    delete_btn = gr.Button("Delete selected video", variant="stop")
                    delete_status = gr.Textbox(label="Delete status", interactive=False)

                refresh_btn.click(
                    fn=_on_refresh_db,
                    outputs=[stats_md, videos_table, segments_table, selected_vid],
                )
                videos_table.select(
                    fn=_on_video_select,
                    inputs=[videos_table],
                    outputs=[selected_vid, segments_table],
                )
                delete_btn.click(
                    fn=_on_delete_video,
                    inputs=[selected_vid, confirm_delete],
                    outputs=[delete_status, videos_table],
                )
                reindex_btn.click(fn=_on_rebuild_indexes, outputs=[reindex_status])

    return demo


def launch(
    db_path: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 7860,
    device: str = "auto",
    share: bool = False,
) -> None:
    """Build the context, build the Gradio app, and start the server."""
    ctx = build_context(db_path, device=device)
    demo = build_app(ctx)
    demo.launch(
        server_name=host,
        server_port=port,
        share=share,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":  # pragma: no cover - convenience entrypoint
    import os

    launch(Path(os.environ.get("VL_DB_PATH", "./video-lance.db")))
