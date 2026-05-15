from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from video_lance import clipper, frames, probe
from video_lance.config import Config
from video_lance.embed_text import E5Embedder
from video_lance.embed_vision import SigLIPEmbedder
from video_lance.models import Transcript, VideoMeta
from video_lance.segmenter import compute_segments
from video_lance.store import (
    SegmentRow,
    StoreTables,
    VideoRow,
    get_video,
    segment_id_for,
    upsert_segments,
    upsert_video,
    video_id_for_path,
)
from video_lance.transcribe import WhisperTranscriber, map_text_to_window


@dataclass
class WorkingSegment:
    """In-memory builder for one segment row. Stages mutate it as they run."""

    video_id: str
    idx: int
    start_s: float
    end_s: float
    keyframe_t_s: float
    text: str = ""
    clip_bytes: bytes | None = None
    keyframe_jpeg: bytes | None = None
    frame_image: Image.Image | None = None
    text_embedding: np.ndarray | None = None
    visual_embedding: np.ndarray | None = None


@dataclass
class PipelineContext:
    """Per-video state that accumulates across stages.

    A fresh context is built for each video processed; stages mutate it in
    place. The UI is expected to support running one stage at a time against
    a partially-populated context, hence `is_ready` on each stage.
    """

    path: Path
    root: Path
    cfg: Config
    tables: StoreTables
    transcriber: WhisperTranscriber
    text_embedder: E5Embedder
    vision_embedder: SigLIPEmbedder
    force: bool = False

    meta: VideoMeta | None = None
    transcript: Transcript | None = None
    segments: list[WorkingSegment] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str | None = None


@dataclass
class StageResult:
    ok: bool
    skipped: bool = False
    note: str | None = None


class Stage(Protocol):
    name: str

    def is_ready(self, ctx: PipelineContext) -> bool: ...
    def run(self, ctx: PipelineContext) -> StageResult: ...


# -- concrete stages ----------------------------------------------------------


class ProbeStage:
    name = "probe"

    def is_ready(self, ctx: PipelineContext) -> bool:
        return ctx.path.exists()

    def run(self, ctx: PipelineContext) -> StageResult:
        ctx.meta = probe.get_meta(ctx.path)

        # Idempotency check (PLAN §5.3): if a row for this video already exists
        # and was built with the same segment knobs, skip everything downstream
        # unless --force.
        if ctx.force:
            return StageResult(ok=True)

        vid = video_id_for_path(ctx.path)
        existing = get_video(ctx.tables, vid)
        if existing is None:
            return StageResult(ok=True)

        same_segment = (
            abs(float(existing["segment_seconds"]) - ctx.cfg.segmentation.segment_seconds) < 1e-6
        )
        same_overlap = (
            abs(float(existing["overlap_seconds"]) - ctx.cfg.segmentation.overlap_seconds) < 1e-6
        )
        if same_segment and same_overlap:
            ctx.skipped = True
            ctx.skip_reason = "already indexed"
            return StageResult(ok=True, skipped=True, note="already indexed")
        return StageResult(ok=True, note="re-segmenting (config changed)")


class TranscribeStage:
    name = "transcribe"

    def is_ready(self, ctx: PipelineContext) -> bool:
        return ctx.meta is not None and not ctx.skipped

    def run(self, ctx: PipelineContext) -> StageResult:
        ctx.transcript = ctx.transcriber.transcribe(ctx.path)
        return StageResult(ok=True)


class SegmentStage:
    name = "segment"

    def is_ready(self, ctx: PipelineContext) -> bool:
        return ctx.meta is not None and ctx.transcript is not None and not ctx.skipped

    def run(self, ctx: PipelineContext) -> StageResult:
        assert ctx.meta is not None and ctx.transcript is not None
        windows = compute_segments(
            ctx.meta.duration_s,
            ctx.cfg.segmentation,
            ctx.transcript.words,
        )
        vid = video_id_for_path(ctx.path)
        out: list[WorkingSegment] = []
        for i, (start, end) in enumerate(windows):
            keyframe_t = start + (end - start) * ctx.cfg.frames.position
            text = map_text_to_window(ctx.transcript, start, end)
            out.append(
                WorkingSegment(
                    video_id=vid,
                    idx=i,
                    start_s=start,
                    end_s=end,
                    keyframe_t_s=keyframe_t,
                    text=text,
                )
            )
        ctx.segments = out
        return StageResult(ok=True, note=f"{len(out)} segments")


class ClipStage:
    name = "clip"

    def is_ready(self, ctx: PipelineContext) -> bool:
        return bool(ctx.segments) and not ctx.skipped

    def run(self, ctx: PipelineContext) -> StageResult:
        for s in ctx.segments:
            s.clip_bytes = clipper.extract_clip_bytes(ctx.path, s.start_s, s.end_s)
        return StageResult(ok=True)


class FrameStage:
    name = "frame"

    def is_ready(self, ctx: PipelineContext) -> bool:
        return bool(ctx.segments) and not ctx.skipped

    def run(self, ctx: PipelineContext) -> StageResult:
        for s in ctx.segments:
            jpeg, image = frames.extract_keyframe(ctx.path, s.keyframe_t_s, ctx.cfg.frames)
            s.keyframe_jpeg = jpeg
            s.frame_image = image
        return StageResult(ok=True)


class EmbedTextStage:
    name = "embed_text"

    def is_ready(self, ctx: PipelineContext) -> bool:
        return bool(ctx.segments) and not ctx.skipped

    def run(self, ctx: PipelineContext) -> StageResult:
        texts = [s.text for s in ctx.segments]
        vecs = ctx.text_embedder.encode_passages(texts)
        if vecs.shape[0] != len(ctx.segments):
            raise RuntimeError(
                f"text embedder returned {vecs.shape[0]} vectors for {len(ctx.segments)} segments"
            )
        for s, v in zip(ctx.segments, vecs, strict=True):
            s.text_embedding = np.asarray(v, dtype=np.float32)
        return StageResult(ok=True)


class EmbedVisionStage:
    name = "embed_vision"

    def is_ready(self, ctx: PipelineContext) -> bool:
        return (
            bool(ctx.segments)
            and not ctx.skipped
            and all(s.frame_image is not None for s in ctx.segments)
        )

    def run(self, ctx: PipelineContext) -> StageResult:
        images: list[Image.Image | bytes] = [
            s.frame_image for s in ctx.segments if s.frame_image is not None
        ]
        vecs = ctx.vision_embedder.encode_images(images)
        if vecs.shape[0] != len(ctx.segments):
            raise RuntimeError(
                f"vision embedder returned {vecs.shape[0]} vectors for "
                f"{len(ctx.segments)} segments"
            )
        for s, v in zip(ctx.segments, vecs, strict=True):
            s.visual_embedding = np.asarray(v, dtype=np.float32)
        return StageResult(ok=True)


class WriteStage:
    name = "write"

    def is_ready(self, ctx: PipelineContext) -> bool:
        if not ctx.segments or ctx.skipped:
            return False
        return all(
            s.clip_bytes is not None
            and s.keyframe_jpeg is not None
            and s.text_embedding is not None
            and s.visual_embedding is not None
            for s in ctx.segments
        )

    def run(self, ctx: PipelineContext) -> StageResult:
        assert ctx.meta is not None
        assert ctx.transcript is not None

        vid = video_id_for_path(ctx.path)
        absolute = ctx.path.resolve()
        try:
            relative = absolute.relative_to(ctx.root.resolve())
        except ValueError:
            relative = Path(ctx.path.name)

        video_row = VideoRow(
            video_id=vid,
            source_path=str(absolute),
            relative_path=str(relative),
            duration_s=ctx.meta.duration_s,
            fps=ctx.meta.fps,
            width=ctx.meta.width,
            height=ctx.meta.height,
            codec=ctx.meta.codec,
            size_bytes=ctx.meta.size_bytes,
            ingested_at=datetime.now(UTC),
            segment_seconds=ctx.cfg.segmentation.segment_seconds,
            overlap_seconds=ctx.cfg.segmentation.overlap_seconds,
            transcript_full=ctx.transcript.full_text,
        )
        upsert_video(ctx.tables, video_row)

        seg_rows = []
        for s in ctx.segments:
            assert s.clip_bytes is not None
            assert s.keyframe_jpeg is not None
            assert s.text_embedding is not None
            assert s.visual_embedding is not None
            seg_rows.append(
                SegmentRow(
                    segment_id=segment_id_for(vid, s.idx),
                    video_id=vid,
                    idx=s.idx,
                    start_s=s.start_s,
                    end_s=s.end_s,
                    keyframe_t_s=s.keyframe_t_s,
                    text=s.text,
                    text_embedding=s.text_embedding.tolist(),
                    visual_embedding=s.visual_embedding.tolist(),
                    clip_bytes=s.clip_bytes,
                    keyframe_jpeg=s.keyframe_jpeg,
                )
            )
        upsert_segments(ctx.tables, seg_rows)
        return StageResult(ok=True, note=f"wrote {len(seg_rows)} segments")


DEFAULT_STAGES: list[Stage] = [
    ProbeStage(),
    TranscribeStage(),
    SegmentStage(),
    ClipStage(),
    FrameStage(),
    EmbedTextStage(),
    EmbedVisionStage(),
    WriteStage(),
]
