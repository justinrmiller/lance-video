from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tests._fakes import fake_text_embedder, fake_transcriber, fake_vision_embedder
from video_lance import store
from video_lance.config import Config, FrameSamplingConfig, SegmentationConfig
from video_lance.models import Transcript, TranscriptWord
from video_lance.stages import (
    DEFAULT_STAGES,
    EmbedTextStage,
    EmbedVisionStage,
    FrameStage,
    PipelineContext,
    ProbeStage,
    SegmentStage,
    TranscribeStage,
    WriteStage,
)
from video_lance.transcribe import map_text_to_window


def _build_ctx(tmp_path: Path, fixture_video: Path) -> PipelineContext:
    cfg = Config(
        segmentation=SegmentationConfig(
            segment_seconds=2.0, overlap_seconds=0.0, merge_short_tail=False
        ),
        frames=FrameSamplingConfig(max_long_edge=128, jpeg_quality=60),
        db_path=tmp_path / "db",
    )
    db = store.connect(cfg.db_path)
    tables = store.ensure_tables(db)
    return PipelineContext(
        path=fixture_video,
        root=fixture_video.parent,
        cfg=cfg,
        tables=tables,
        transcriber=fake_transcriber(),
        text_embedder=fake_text_embedder(),
        vision_embedder=fake_vision_embedder(),
    )


def test_default_stages_in_documented_order() -> None:
    names = [s.name for s in DEFAULT_STAGES]
    assert names == [
        "probe",
        "transcribe",
        "segment",
        "clip",
        "frame",
        "embed_text",
        "embed_vision",
        "write",
    ]


def test_probe_then_segment_populates_context(tmp_path: Path, fixture_video: Path) -> None:
    ctx = _build_ctx(tmp_path, fixture_video)
    ProbeStage().run(ctx)
    assert ctx.meta is not None
    assert ctx.meta.duration_s == pytest.approx(10.0, abs=0.1)

    TranscribeStage().run(ctx)
    assert ctx.transcript is not None
    assert len(ctx.transcript.words) == 5  # fake transcriber emits 5 words

    SegmentStage().run(ctx)
    assert len(ctx.segments) == 5
    assert ctx.segments[0].start_s == 0.0
    # Each working segment carries the text mapped from the transcript.
    assert any(s.text for s in ctx.segments)


def test_probe_idempotency_flag_short_circuits_downstream(
    tmp_path: Path, fixture_video: Path
) -> None:
    ctx = _build_ctx(tmp_path, fixture_video)
    # Pre-populate the table with a matching video row.
    from datetime import UTC, datetime

    from video_lance.store import VideoRow, upsert_video, video_id_for_path

    upsert_video(
        ctx.tables,
        VideoRow(
            video_id=video_id_for_path(fixture_video),
            source_path=str(fixture_video.resolve()),
            relative_path=fixture_video.name,
            duration_s=10.0,
            fps=30.0,
            width=320,
            height=240,
            codec="h264",
            size_bytes=fixture_video.stat().st_size,
            ingested_at=datetime.now(UTC),
            segment_seconds=ctx.cfg.segmentation.segment_seconds,
            overlap_seconds=ctx.cfg.segmentation.overlap_seconds,
            transcript_full="",
        ),
    )

    result = ProbeStage().run(ctx)
    assert result.skipped is True
    assert ctx.skipped is True

    # All downstream stages should report not-ready while skipped is set.
    assert not TranscribeStage().is_ready(ctx)
    assert not SegmentStage().is_ready(ctx)
    assert not WriteStage().is_ready(ctx)


def test_force_overrides_idempotency(tmp_path: Path, fixture_video: Path) -> None:
    ctx = _build_ctx(tmp_path, fixture_video)
    ctx.force = True

    from datetime import UTC, datetime

    from video_lance.store import VideoRow, upsert_video, video_id_for_path

    upsert_video(
        ctx.tables,
        VideoRow(
            video_id=video_id_for_path(fixture_video),
            source_path=str(fixture_video.resolve()),
            relative_path=fixture_video.name,
            duration_s=10.0,
            fps=30.0,
            width=320,
            height=240,
            codec="h264",
            size_bytes=fixture_video.stat().st_size,
            ingested_at=datetime.now(UTC),
            segment_seconds=ctx.cfg.segmentation.segment_seconds,
            overlap_seconds=ctx.cfg.segmentation.overlap_seconds,
            transcript_full="",
        ),
    )

    result = ProbeStage().run(ctx)
    assert not result.skipped
    assert not ctx.skipped


def test_embed_stages_require_inputs(tmp_path: Path, fixture_video: Path) -> None:
    ctx = _build_ctx(tmp_path, fixture_video)
    # No segments yet → embed stages are not ready.
    assert not EmbedTextStage().is_ready(ctx)
    assert not EmbedVisionStage().is_ready(ctx)

    ProbeStage().run(ctx)
    TranscribeStage().run(ctx)
    SegmentStage().run(ctx)
    # Segments exist but frame images don't yet → vision stage still not ready.
    assert EmbedTextStage().is_ready(ctx)  # only needs text
    assert not EmbedVisionStage().is_ready(ctx)  # needs frame images

    FrameStage().run(ctx)
    assert EmbedVisionStage().is_ready(ctx)


def test_write_stage_requires_all_segment_fields(tmp_path: Path, fixture_video: Path) -> None:
    ctx = _build_ctx(tmp_path, fixture_video)
    for stage in DEFAULT_STAGES[:-1]:
        stage.run(ctx)
    # All upstream stages have populated the segment fields.
    assert WriteStage().is_ready(ctx)
    WriteStage().run(ctx)
    assert ctx.tables.segments.count_rows() == 5


# -- map_text_to_window unit tests --------------------------------------------


def test_map_text_to_window_overlap() -> None:
    tr = Transcript(
        words=[
            TranscriptWord(word="alpha", start=0.0, end=1.0),
            TranscriptWord(word="beta.", start=1.5, end=2.5),
            TranscriptWord(word="gamma", start=3.0, end=4.0),
        ]
    )
    assert map_text_to_window(tr, 0.0, 2.0) == "alpha beta."
    assert map_text_to_window(tr, 1.0, 3.0) == "beta."
    assert map_text_to_window(tr, 4.0, 5.0) == ""


def test_map_text_to_window_strips_whitespace() -> None:
    tr = Transcript(
        words=[
            TranscriptWord(word="  hi  ", start=0.0, end=0.5),
            TranscriptWord(word=" ", start=0.5, end=1.0),
        ]
    )
    assert map_text_to_window(tr, 0.0, 1.0) == "hi"


# -- using ingest root for completeness ---------------------------------------


@pytest.fixture
def ingest_root(tmp_path: Path, fixture_video: Path) -> Path:
    root = tmp_path / "stages_root"
    root.mkdir()
    shutil.copy(fixture_video, root / "vid.mp4")
    return root
