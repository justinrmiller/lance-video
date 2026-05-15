from __future__ import annotations

import io
import shutil
from pathlib import Path

import pytest
from PIL import Image

from tests._fakes import fake_text_embedder, fake_transcriber, fake_vision_embedder
from video_lance import store
from video_lance.config import Config, FrameSamplingConfig, SegmentationConfig
from video_lance.pipeline import process_video
from video_lance.ui_app import (
    DISCOVER_COLUMNS,
    SEGMENTS_COLUMNS,
    VIDEOS_COLUMNS,
    AppContext,
    _caption,
    _hit_to_state,
    build_context,
    db_stats_markdown,
    delete_video_action,
    discover_for_table,
    list_segments_for_video,
    list_videos,
    play_clip,
    rebuild_indexes_action,
    run_ingest_streaming,
    run_search,
)


@pytest.fixture
def ctx(tmp_path: Path, fixture_video: Path) -> AppContext:
    """Spin up a real LanceDB-backed context with 5 segments and fake
    encoders. We bypass `build_context()` (which would download real models)
    by constructing `AppContext` directly with the test fakes."""
    cfg = Config(
        segmentation=SegmentationConfig(segment_seconds=2.0, merge_short_tail=False),
        frames=FrameSamplingConfig(max_long_edge=128, jpeg_quality=60),
        db_path=tmp_path / "db",
    )
    db = store.connect(cfg.db_path)
    tables = store.ensure_tables(db)
    store.set_embedding_models(
        tables, text_embed_model="fake-e5", vision_embed_model="fake-siglip"
    )
    process_video(
        fixture_video,
        fixture_video.parent,
        cfg,
        tables,
        transcriber=fake_transcriber(),
        text_embedder=fake_text_embedder(),
        vision_embedder=fake_vision_embedder(),
    )
    return AppContext(
        db_path=cfg.db_path,
        tables=tables,
        text_embedder=fake_text_embedder(),
        vision_embedder=fake_vision_embedder(),
    )


# -- run_search --------------------------------------------------------------


def test_run_search_text(ctx: AppContext) -> None:
    # Build FTS first so the hybrid path runs both legs.
    from video_lance.search import ensure_indexes

    ensure_indexes(ctx.tables)

    gallery, raw = run_search(ctx, "red.", "text", None, 5, "", 0.4)
    assert len(gallery) <= 5
    assert len(raw) == len(gallery)
    for img, caption in gallery:
        assert isinstance(img, Image.Image)
        assert img.size[0] > 0 and img.size[1] > 0
        assert caption.startswith(tuple(f"{i}. [" for i in range(1, 6)))
    for h in raw:
        assert set(h.keys()) >= {
            "segment_id",
            "video_id",
            "start_s",
            "end_s",
            "score",
            "source_path",
            "relative_path",
        }


def test_run_search_visual_text_query(ctx: AppContext) -> None:
    gallery, raw = run_search(ctx, "anything", "visual", None, 3, "", 0.4)
    assert len(gallery) == 3
    assert len(raw) == 3


def test_run_search_visual_with_image(ctx: AppContext) -> None:
    img = Image.new("RGB", (32, 32), (255, 0, 0))
    gallery, raw = run_search(ctx, "", "visual", img, 3, "", 0.4)
    assert len(gallery) == 3
    assert len(raw) == 3


def test_run_search_multi_blends(ctx: AppContext) -> None:
    gallery, raw = run_search(ctx, "anything", "multi", None, 4, "", 0.4)
    assert len(gallery) == 4
    for h in raw:
        # Multi mode tags both sources in the components dict.
        comps = h["components"]
        assert "text" in comps or "visual" in comps


def test_run_search_unknown_mode_rejected(ctx: AppContext) -> None:
    with pytest.raises(ValueError, match="unknown mode"):
        run_search(ctx, "x", "bogus", None, 5, "", 0.4)


def test_run_search_text_requires_query(ctx: AppContext) -> None:
    with pytest.raises(ValueError, match="text mode requires"):
        run_search(ctx, "", "text", None, 5, "", 0.4)


def test_run_search_visual_requires_query_or_image(ctx: AppContext) -> None:
    with pytest.raises(ValueError, match="visual mode requires"):
        run_search(ctx, "", "visual", None, 5, "", 0.4)


def test_run_search_multi_requires_query(ctx: AppContext) -> None:
    with pytest.raises(ValueError, match="multi mode requires"):
        run_search(ctx, "", "multi", None, 5, "", 0.4)


def test_run_search_sql_filter_passes_through(ctx: AppContext) -> None:
    gallery, raw = run_search(ctx, "x", "visual", None, 10, "idx >= 3", 0.4)
    # Filter restricts to segments idx 3 and 4 → 2 rows total.
    assert len(raw) <= 2
    for h in raw:
        assert h["idx"] >= 3


# -- play_clip ---------------------------------------------------------------


def test_play_clip_returns_valid_mp4_path(ctx: AppContext) -> None:
    _, raw = run_search(ctx, "x", "visual", None, 3, "", 0.4)
    assert raw, "fixture should have produced hits"
    path = play_clip(ctx, raw, 0)
    assert path is not None
    p = Path(path)
    assert p.exists()
    assert p.suffix == ".mp4"
    data = p.read_bytes()
    assert b"ftyp" in data[:64]  # valid MP4 container marker


def test_play_clip_out_of_range_returns_none(ctx: AppContext) -> None:
    _, raw = run_search(ctx, "x", "visual", None, 3, "", 0.4)
    assert play_clip(ctx, raw, -1) is None
    assert play_clip(ctx, raw, 999) is None


def test_play_clip_no_selection_returns_none(ctx: AppContext) -> None:
    _, raw = run_search(ctx, "x", "visual", None, 3, "", 0.4)
    assert play_clip(ctx, raw, None) is None
    assert play_clip(ctx, [], 0) is None


# -- helpers -----------------------------------------------------------------


def test_hit_to_state_round_trip_keys() -> None:
    from video_lance.search import SearchHit

    hit = SearchHit(
        segment_id="x:000000",
        video_id="x",
        source_path="/a.mp4",
        relative_path="a.mp4",
        idx=0,
        start_s=0.0,
        end_s=2.0,
        text="hi",
        score=0.5,
        components={"text": 0.5},
    )
    state = _hit_to_state(hit)
    assert state["segment_id"] == "x:000000"
    assert state["components"] == {"text": 0.5}
    # Must be JSON/pickle clean (no numpy / non-primitive types).
    import json

    json.dumps(state)


def test_caption_format() -> None:
    from video_lance.search import SearchHit

    hit = SearchHit(
        segment_id="x:000000",
        video_id="x",
        source_path="/a.mp4",
        relative_path="my/video.mp4",
        idx=0,
        start_s=754.5,
        end_s=784.5,
        text="hello world",
        score=0.83,
    )
    cap = _caption(1, hit)
    assert cap.startswith("1. [0.830]")
    assert "my/video.mp4" in cap
    assert "00:12:34" in cap  # time_range formatting


# -- build_context behavior --------------------------------------------------


def test_build_context_rejects_missing_db(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_context(tmp_path / "no-such-db")


# -- gallery thumbnails resolve to real JPEGs --------------------------------


def test_run_search_thumbnails_are_real_jpegs(ctx: AppContext) -> None:
    gallery, _ = run_search(ctx, "x", "visual", None, 3, "", 0.4)
    for img, _cap in gallery:
        # Round-trip through PNG to make sure the image actually has pixel data.
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        decoded = Image.open(io.BytesIO(buf.getvalue()))
        decoded.verify()


# -- Gradio build is importable but lazy ------------------------------------


def test_build_app_runs(ctx: AppContext) -> None:
    """`build_app` should import gradio + return a Blocks without crashing.
    We don't launch a server, just verify construction."""
    from video_lance.ui_app import build_app

    demo = build_app(ctx)
    # Gradio Blocks expose `.fns` (a registry of event handlers) and `.title`.
    assert demo is not None
    assert hasattr(demo, "launch")
    # Must have wired the Search / Ingest / Database tabs — at minimum the
    # search-button click, gallery-select, discover, run-ingest, refresh,
    # video-row-select, delete, and reindex handlers.
    n_handlers = len(getattr(demo, "fns", []) or [])
    assert n_handlers >= 8, f"expected at least 8 event handlers, got {n_handlers}"


# ===========================================================================
# Database view
# ===========================================================================


def test_list_videos_shape_and_segment_count(ctx: AppContext) -> None:
    rows = list_videos(ctx.tables)
    assert len(rows) == 1
    row = rows[0]
    for col in VIDEOS_COLUMNS:
        assert col in row, f"missing column {col}"
    # The fixture video is 10 s segmented at 2 s with no tail merge → 5 segments.
    assert row["segments"] == 5
    assert row["duration_s"] == pytest.approx(10.0, abs=0.1)
    assert row["width"] == 320 and row["height"] == 240
    assert row["size_mb"] > 0


def test_list_segments_for_video(ctx: AppContext) -> None:
    vids = list_videos(ctx.tables)
    rows = list_segments_for_video(ctx.tables, vids[0]["video_id"])
    assert len(rows) == 5
    for r in rows:
        for col in SEGMENTS_COLUMNS:
            assert col in r
    # Ordered by idx ascending.
    assert [r["idx"] for r in rows] == [0, 1, 2, 3, 4]


def test_list_segments_for_unknown_video_id(ctx: AppContext) -> None:
    assert list_segments_for_video(ctx.tables, "no-such-id") == []


def test_db_stats_markdown_contains_key_facts(ctx: AppContext) -> None:
    md = db_stats_markdown(ctx.tables, ctx.db_path)
    assert "videos:" in md
    assert "segments:" in md
    assert "fake-e5" in md
    assert "fake-siglip" in md


def test_delete_video_action_requires_confirm(ctx: AppContext) -> None:
    vid = list_videos(ctx.tables)[0]["video_id"]
    status = delete_video_action(ctx.tables, vid, confirm=False)
    assert "confirmation required" in status.lower()
    # Row must still be there.
    assert ctx.tables.videos.count_rows() == 1


def test_delete_video_action_with_confirm_removes_row_and_segments(
    ctx: AppContext,
) -> None:
    vid = list_videos(ctx.tables)[0]["video_id"]
    assert ctx.tables.videos.count_rows() == 1
    assert ctx.tables.segments.count_rows() == 5

    status = delete_video_action(ctx.tables, vid, confirm=True)
    assert vid in status

    assert ctx.tables.videos.count_rows() == 0
    assert ctx.tables.segments.count_rows() == 0


def test_delete_video_action_unknown_id(ctx: AppContext) -> None:
    status = delete_video_action(ctx.tables, "no-such-id-123", confirm=True)
    assert "no row" in status.lower()


def test_delete_video_action_empty_id(ctx: AppContext) -> None:
    status = delete_video_action(ctx.tables, "", confirm=True)
    assert "no video selected" in status.lower()


def test_rebuild_indexes_action_runs(ctx: AppContext) -> None:
    status = rebuild_indexes_action(ctx.tables)
    assert "fts_text" in status
    assert "vec_text_embedding" in status
    assert "vec_visual_embed" in status


# ===========================================================================
# Ingest view
# ===========================================================================


def test_discover_for_table_finds_videos(
    tmp_path: Path, fixture_video: Path
) -> None:
    root = tmp_path / "src"
    root.mkdir()
    shutil.copy(fixture_video, root / "a.mp4")
    (root / "sub").mkdir()
    shutil.copy(fixture_video, root / "sub" / "b.mp4")

    rows = discover_for_table(root)
    assert len(rows) == 2
    for r in rows:
        for col in DISCOVER_COLUMNS:
            assert col in r
        assert r["size_mb"] > 0


def test_discover_for_table_respects_exclude(
    tmp_path: Path, fixture_video: Path
) -> None:
    root = tmp_path / "src"
    root.mkdir()
    shutil.copy(fixture_video, root / "keep.mp4")
    shutil.copy(fixture_video, root / "drop_me.mp4")

    rows = discover_for_table(root, exclude=("*drop*",))
    names = [r["name"] for r in rows]
    assert "keep.mp4" in names
    assert "drop_me.mp4" not in names


def test_discover_for_table_missing_root(tmp_path: Path) -> None:
    assert discover_for_table(tmp_path / "no-such") == []


def test_discover_for_table_file_argument(tmp_path: Path, fixture_video: Path) -> None:
    """A single file path: `discovery.walk` already handles it; we pass it
    through. Useful for the UI when the textbox points at one file rather
    than a directory."""
    target = tmp_path / "one.mp4"
    shutil.copy(fixture_video, target)
    rows = discover_for_table(target)
    assert len(rows) == 1
    assert rows[0]["name"] == "one.mp4"


# -- ingest streaming --------------------------------------------------------


@pytest.fixture
def fresh_ctx(tmp_path: Path) -> AppContext:
    """Empty DB + fake encoders + a fake-transcriber wired into get_transcriber.

    Unlike `ctx`, this fixture starts with zero rows so the streaming ingest
    populates the DB during the test."""
    db_path = tmp_path / "db"
    db = store.connect(db_path)
    tables = store.ensure_tables(db)
    store.set_embedding_models(
        tables, text_embed_model="fake-e5", vision_embed_model="fake-siglip"
    )
    ctx = AppContext(
        db_path=db_path,
        tables=tables,
        text_embedder=fake_text_embedder(),
        vision_embedder=fake_vision_embedder(),
        device="cpu",
        whisper_model="fake-whisper",
    )
    # Bypass the real Whisper loader — return our fake transcriber instead.
    ctx.get_transcriber = lambda: fake_transcriber()  # type: ignore[method-assign]
    return ctx


def test_run_ingest_streaming_processes_each_video(
    fresh_ctx: AppContext, tmp_path: Path, fixture_video: Path
) -> None:
    root = tmp_path / "ingest"
    root.mkdir()
    shutil.copy(fixture_video, root / "a.mp4")
    shutil.copy(fixture_video, root / "b.mp4")

    progress_values: list[float] = []
    last_log = ""
    for prog, log in run_ingest_streaming(
        fresh_ctx,
        root,
        segment_seconds=2.0,
        merge_short_tail=False,
        frame_max_long_edge=128,
        frame_jpeg_quality=60,
    ):
        progress_values.append(prog)
        last_log = log

    # Final progress is 1.0, and there's a yield per video plus the bookend pair.
    assert progress_values[-1] == 1.0
    assert progress_values[0] == 0.0
    # Two videos written.
    assert fresh_ctx.tables.videos.count_rows() == 2
    assert fresh_ctx.tables.segments.count_rows() == 10
    assert "succeeded=2" in last_log


def test_run_ingest_streaming_no_match(
    fresh_ctx: AppContext, tmp_path: Path
) -> None:
    empty = tmp_path / "nothing"
    empty.mkdir()
    yields = list(run_ingest_streaming(fresh_ctx, empty, segment_seconds=2.0))
    assert yields
    final_prog, final_log = yields[-1]
    assert final_prog == 1.0
    assert "no videos matched" in final_log


def test_run_ingest_streaming_missing_root(fresh_ctx: AppContext, tmp_path: Path) -> None:
    yields = list(run_ingest_streaming(fresh_ctx, tmp_path / "absent", segment_seconds=2.0))
    final_prog, final_log = yields[-1]
    assert final_prog == 1.0
    assert "does not exist" in final_log


def test_run_ingest_streaming_skip_on_re_run(
    fresh_ctx: AppContext, tmp_path: Path, fixture_video: Path
) -> None:
    root = tmp_path / "ingest"
    root.mkdir()
    shutil.copy(fixture_video, root / "a.mp4")

    # First run writes the row.
    for _ in run_ingest_streaming(
        fresh_ctx, root, segment_seconds=2.0, merge_short_tail=False
    ):
        pass
    assert fresh_ctx.tables.segments.count_rows() == 5

    # Second run with same config skips.
    last_log = ""
    for _prog, log in run_ingest_streaming(
        fresh_ctx, root, segment_seconds=2.0, merge_short_tail=False
    ):
        last_log = log
    assert "skip" in last_log.lower()
    assert "skipped=1" in last_log


def test_run_ingest_streaming_force_reingests(
    fresh_ctx: AppContext, tmp_path: Path, fixture_video: Path
) -> None:
    root = tmp_path / "ingest"
    root.mkdir()
    shutil.copy(fixture_video, root / "a.mp4")

    for _ in run_ingest_streaming(
        fresh_ctx, root, segment_seconds=2.0, merge_short_tail=False
    ):
        pass
    last_log = ""
    for _prog, log in run_ingest_streaming(
        fresh_ctx, root, segment_seconds=2.0, merge_short_tail=False, force=True
    ):
        last_log = log
    assert "succeeded=1" in last_log


# -- AppContext.get_transcriber ----------------------------------------------


def test_app_context_carries_device_and_whisper_model() -> None:
    """The default field values shouldn't require touching the real Whisper
    loader unless `.get_transcriber()` is actually called."""
    db_path = Path("/tmp/does-not-exist-for-this-assertion")
    fake_tables = object()  # type: ignore[assignment]
    ctx = AppContext(
        db_path=db_path,
        tables=fake_tables,  # type: ignore[arg-type]
        text_embedder=fake_text_embedder(),
        vision_embedder=fake_vision_embedder(),
        device="cuda",
        whisper_model="medium.en",
    )
    assert ctx.device == "cuda"
    assert ctx.whisper_model == "medium.en"
