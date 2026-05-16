from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from tests._fakes import fake_text_embedder, fake_transcriber, fake_vision_embedder
from video_lance import store
from video_lance.config import (
    Config,
    FrameSamplingConfig,
    SegmentationConfig,
)
from video_lance.pipeline import process_directory, process_video
from video_lance.schema import TEXT_EMBED_DIM, VISION_EMBED_DIM


@pytest.fixture
def ingest_root(tmp_path: Path, fixture_video: Path) -> Path:
    """Mini directory tree containing the fixture video at two levels."""
    root = tmp_path / "ingest_root"
    (root / "sub").mkdir(parents=True)
    shutil.copy(fixture_video, root / "top.mp4")
    shutil.copy(fixture_video, root / "sub" / "nested.mp4")
    return root


def _build_config(tmp_path: Path, *, segment_seconds: float = 2.0) -> Config:
    return Config(
        segmentation=SegmentationConfig(
            segment_seconds=segment_seconds,
            overlap_seconds=0.0,
            merge_short_tail=False,
        ),
        frames=FrameSamplingConfig(max_long_edge=128, jpeg_quality=60),
        db_path=tmp_path / "db",
    )


def test_smoke_process_video_writes_segments_with_correct_dims(
    tmp_path: Path, fixture_video: Path
) -> None:
    """End-to-end on the fixture video: every stage runs, both vector columns
    are populated with float32 vectors of the right dim, and the blob columns
    round-trip via `read_segment_blob`. This is PLAN §10 Session 5's
    acceptance check."""
    cfg = _build_config(tmp_path, segment_seconds=2.0)
    db = store.connect(cfg.db_path)
    tables = store.ensure_tables(db)

    result = process_video(
        fixture_video,
        fixture_video.parent,
        cfg,
        tables,
        transcriber=fake_transcriber(),
        text_embedder=fake_text_embedder(),
        vision_embedder=fake_vision_embedder(),
    )

    assert result.ok
    assert not result.skipped
    assert result.segments_written == 5  # 10s fixture / 2s segments

    vid = store.video_id_for_path(fixture_video)
    video_row = store.get_video(tables, vid)
    assert video_row is not None
    assert video_row["duration_s"] == pytest.approx(10.0, abs=0.1)
    assert video_row["width"] == 320 and video_row["height"] == 240

    segs = store.get_segments_for_video(tables, vid)
    assert len(segs) == 5
    for s in segs:
        assert len(s["text_embedding"]) == TEXT_EMBED_DIM
        assert len(s["visual_embedding"]) == VISION_EMBED_DIM
        # All vectors must be finite L2-unit-norm (the wrappers normalize).
        text_vec = np.asarray(s["text_embedding"], dtype=np.float32)
        visual_vec = np.asarray(s["visual_embedding"], dtype=np.float32)
        assert np.isfinite(text_vec).all() and np.isfinite(visual_vec).all()
        assert np.linalg.norm(text_vec) == pytest.approx(1.0, abs=1e-4)
        assert np.linalg.norm(visual_vec) == pytest.approx(1.0, abs=1e-4)

    # Blob round-trip on a real ffmpeg-extracted clip + keyframe.
    sid = segs[0]["segment_id"]
    clip = store.read_segment_blob(tables, sid, "clip_bytes")
    assert clip.startswith(b"\x00\x00\x00") or b"ftyp" in clip[:64]
    jpeg = store.read_segment_blob(tables, sid, "keyframe_jpeg")
    assert jpeg.startswith(b"\xff\xd8")  # JPEG SOI


def test_idempotent_skip_when_segment_config_matches(
    tmp_path: Path, fixture_video: Path
) -> None:
    cfg = _build_config(tmp_path, segment_seconds=2.0)
    db = store.connect(cfg.db_path)
    tables = store.ensure_tables(db)

    first = process_video(
        fixture_video,
        fixture_video.parent,
        cfg,
        tables,
        transcriber=fake_transcriber(),
        text_embedder=fake_text_embedder(),
        vision_embedder=fake_vision_embedder(),
    )
    assert first.ok and not first.skipped and first.segments_written == 5

    # Same config → must skip.
    second = process_video(
        fixture_video,
        fixture_video.parent,
        cfg,
        tables,
        transcriber=fake_transcriber(),
        text_embedder=fake_text_embedder(),
        vision_embedder=fake_vision_embedder(),
    )
    assert second.ok and second.skipped and second.segments_written == 0
    assert second.skip_reason == "already indexed"


def test_reingest_with_different_segment_seconds_replaces(
    tmp_path: Path, fixture_video: Path
) -> None:
    db_path = tmp_path / "db"
    db = store.connect(db_path)
    tables = store.ensure_tables(db)

    cfg2 = Config(
        segmentation=SegmentationConfig(segment_seconds=2.0, merge_short_tail=False),
        frames=FrameSamplingConfig(max_long_edge=128, jpeg_quality=60),
        db_path=db_path,
    )
    process_video(
        fixture_video,
        fixture_video.parent,
        cfg2,
        tables,
        transcriber=fake_transcriber(),
        text_embedder=fake_text_embedder(),
        vision_embedder=fake_vision_embedder(),
    )

    cfg5 = Config(
        segmentation=SegmentationConfig(segment_seconds=5.0, merge_short_tail=False),
        frames=FrameSamplingConfig(max_long_edge=128, jpeg_quality=60),
        db_path=db_path,
    )
    result = process_video(
        fixture_video,
        fixture_video.parent,
        cfg5,
        tables,
        transcriber=fake_transcriber(),
        text_embedder=fake_text_embedder(),
        vision_embedder=fake_vision_embedder(),
    )
    assert result.ok and not result.skipped

    vid = store.video_id_for_path(fixture_video)
    segs = store.get_segments_for_video(tables, vid)
    assert len(segs) == 2  # 10s / 5s = 2 segments now


def test_force_re_ingests_even_if_indexed(tmp_path: Path, fixture_video: Path) -> None:
    cfg = _build_config(tmp_path, segment_seconds=2.0)
    db = store.connect(cfg.db_path)
    tables = store.ensure_tables(db)
    args = dict(
        transcriber=fake_transcriber(),
        text_embedder=fake_text_embedder(),
        vision_embedder=fake_vision_embedder(),
    )
    process_video(fixture_video, fixture_video.parent, cfg, tables, **args)
    forced = process_video(
        fixture_video, fixture_video.parent, cfg, tables, force=True, **args
    )
    assert not forced.skipped
    assert forced.segments_written == 5


def test_process_directory_handles_multiple_videos(ingest_root: Path, tmp_path: Path) -> None:
    cfg = _build_config(tmp_path, segment_seconds=2.0)
    db = store.connect(cfg.db_path)
    tables = store.ensure_tables(db)

    batch = process_directory(
        ingest_root,
        cfg,
        tables,
        transcriber=fake_transcriber(),
        text_embedder=fake_text_embedder(),
        vision_embedder=fake_vision_embedder(),
    )

    assert len(batch.discovered) == 2
    assert batch.succeeded == 2
    assert batch.skipped == 0
    assert batch.failed == 0
    assert batch.written == 10  # 5 segments * 2 videos
    assert tables.videos.count_rows() == 2
    assert tables.segments.count_rows() == 10


def test_process_directory_empty_root(tmp_path: Path) -> None:
    cfg = _build_config(tmp_path)
    db = store.connect(cfg.db_path)
    tables = store.ensure_tables(db)

    empty = tmp_path / "nothing"
    empty.mkdir()
    batch = process_directory(
        empty,
        cfg,
        tables,
        transcriber=fake_transcriber(),
        text_embedder=fake_text_embedder(),
        vision_embedder=fake_vision_embedder(),
    )
    assert batch.discovered == []
    assert batch.results == []
    assert batch.written == 0


def test_failure_is_isolated_per_video(tmp_path: Path, fixture_video: Path) -> None:
    """A broken file in the batch should not stop the others."""
    root = tmp_path / "mixed"
    root.mkdir()
    shutil.copy(fixture_video, root / "good.mp4")
    (root / "broken.mp4").write_bytes(b"this is not actually a video")

    cfg = _build_config(tmp_path, segment_seconds=2.0)
    db = store.connect(cfg.db_path)
    tables = store.ensure_tables(db)

    batch = process_directory(
        root,
        cfg,
        tables,
        transcriber=fake_transcriber(),
        text_embedder=fake_text_embedder(),
        vision_embedder=fake_vision_embedder(),
    )

    assert batch.succeeded == 1
    assert batch.failed == 1
    assert batch.written == 5
    failed = next(r for r in batch.results if not r.ok)
    assert failed.error is not None


# -- auto-index after ingest ------------------------------------------------


def _has_fts_on_text(tables: store.StoreTables) -> bool:
    for idx in tables.segments.list_indices():
        if "text" in (getattr(idx, "columns", None) or []):
            return True
    return False


def test_process_directory_auto_indexes_by_default(
    ingest_root: Path, tmp_path: Path
) -> None:
    """After a successful batch, the FTS index on `text` must exist — without
    it, `search_text` falls back to vector-only with a noisy warning. The
    PLAN's hybrid retrieval depends on FTS being built."""
    cfg = _build_config(tmp_path, segment_seconds=2.0)
    db = store.connect(cfg.db_path)
    tables = store.ensure_tables(db)

    batch = process_directory(
        ingest_root,
        cfg,
        tables,
        transcriber=fake_transcriber(),
        text_embedder=fake_text_embedder(),
        vision_embedder=fake_vision_embedder(),
    )
    assert batch.succeeded == 2
    assert _has_fts_on_text(tables), "auto_index should have built FTS on `text`"


def test_process_directory_auto_index_disabled(
    ingest_root: Path, tmp_path: Path
) -> None:
    cfg = _build_config(tmp_path, segment_seconds=2.0)
    db = store.connect(cfg.db_path)
    tables = store.ensure_tables(db)

    process_directory(
        ingest_root,
        cfg,
        tables,
        transcriber=fake_transcriber(),
        text_embedder=fake_text_embedder(),
        vision_embedder=fake_vision_embedder(),
        auto_index=False,
    )
    assert not _has_fts_on_text(tables)


def test_process_directory_no_successes_skips_index(
    tmp_path: Path
) -> None:
    """If the batch wrote nothing (empty directory, all failed), there's no
    reason to spin up the FTS builder."""
    empty = tmp_path / "empty"
    empty.mkdir()
    cfg = _build_config(tmp_path, segment_seconds=2.0)
    db = store.connect(cfg.db_path)
    tables = store.ensure_tables(db)

    process_directory(
        empty,
        cfg,
        tables,
        transcriber=fake_transcriber(),
        text_embedder=fake_text_embedder(),
        vision_embedder=fake_vision_embedder(),
    )
    assert not _has_fts_on_text(tables)


def test_search_text_after_ingest_uses_fts_leg(
    ingest_root: Path, tmp_path: Path
) -> None:
    """End-to-end: after `process_directory`, `search_text` should return
    hits that record an `fts` component (the RRF blend of vector + FTS), not
    fall back to vector-only with the warning the user reported."""
    from video_lance.search import search_text

    cfg = _build_config(tmp_path, segment_seconds=2.0)
    db = store.connect(cfg.db_path)
    tables = store.ensure_tables(db)

    process_directory(
        ingest_root,
        cfg,
        tables,
        transcriber=fake_transcriber(),
        text_embedder=fake_text_embedder(),
        vision_embedder=fake_vision_embedder(),
    )

    hits = search_text(tables, fake_text_embedder(), "red.", limit=5)
    assert hits
    # At least one hit should record an `fts` contribution → hybrid leg
    # actually fired, no fallback. (Individual hits that only appeared in
    # the vector leg legitimately won't have an `fts` component; what
    # matters is that the FTS query *ran*.)
    assert any("fts" in h.components for h in hits)
