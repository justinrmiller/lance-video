from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from video_lance import discovery
from video_lance.config import Config
from video_lance.embed_text import E5Embedder
from video_lance.embed_vision import SigLIPEmbedder
from video_lance.stages import (
    DEFAULT_STAGES,
    PipelineContext,
    Stage,
)
from video_lance.store import StoreTables
from video_lance.transcribe import WhisperTranscriber


@dataclass
class ProcessResult:
    path: Path
    ok: bool
    skipped: bool
    segments_written: int
    error: str | None = None
    skip_reason: str | None = None


@dataclass
class BatchResult:
    root: Path
    discovered: list[Path] = field(default_factory=list)
    results: list[ProcessResult] = field(default_factory=list)

    @property
    def written(self) -> int:
        return sum(r.segments_written for r in self.results)

    @property
    def succeeded(self) -> int:
        return sum(1 for r in self.results if r.ok and not r.skipped)

    @property
    def skipped(self) -> int:
        return sum(1 for r in self.results if r.skipped)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.ok)


def process_video(
    path: Path,
    root: Path,
    cfg: Config,
    tables: StoreTables,
    *,
    transcriber: WhisperTranscriber,
    text_embedder: E5Embedder,
    vision_embedder: SigLIPEmbedder,
    force: bool = False,
    stages: list[Stage] | None = None,
) -> ProcessResult:
    """Run every stage in order for one video.

    Stages mutate a fresh `PipelineContext` in place. The first stage that
    isn't ready (`is_ready` returns False) ends the run for this video — for
    a typical ingest that means `ProbeStage` flagged the video as already
    indexed and downstream stages have nothing to do.
    """
    ctx = PipelineContext(
        path=path,
        root=root,
        cfg=cfg,
        tables=tables,
        transcriber=transcriber,
        text_embedder=text_embedder,
        vision_embedder=vision_embedder,
        force=force,
    )

    chain = stages if stages is not None else DEFAULT_STAGES
    try:
        for stage in chain:
            if not stage.is_ready(ctx):
                # Probe sets `skipped=True` for the idempotency path; treat it
                # as success-with-no-write. Any other unready stage means the
                # previous one short-circuited, which is also fine.
                break
            stage.run(ctx)
    except Exception as exc:  # noqa: BLE001 - we surface the error per-video
        return ProcessResult(
            path=path,
            ok=False,
            skipped=False,
            segments_written=0,
            error=f"{type(exc).__name__}: {exc}",
        )

    return ProcessResult(
        path=path,
        ok=True,
        skipped=ctx.skipped,
        segments_written=0 if ctx.skipped else len(ctx.segments),
        skip_reason=ctx.skip_reason,
    )


def process_directory(
    root: Path,
    cfg: Config,
    tables: StoreTables,
    *,
    transcriber: WhisperTranscriber,
    text_embedder: E5Embedder,
    vision_embedder: SigLIPEmbedder,
    force: bool = False,
) -> BatchResult:
    """Discover videos under `root` and process each.

    Sequential for now; PLAN §6 calls for a `ProcessPoolExecutor` keyed on
    `--workers`. That requires pickleable models, which our embedder
    instances aren't — adding it sensibly is a follow-up. Threads aren't a
    useful substitute here because the model encode calls would still
    serialize on the GIL/CUDA lock.
    """
    discovered = discovery.walk(root, include=cfg.include, exclude=cfg.exclude)
    batch = BatchResult(root=root, discovered=list(discovered))

    for path in discovered:
        result = process_video(
            path,
            root,
            cfg,
            tables,
            transcriber=transcriber,
            text_embedder=text_embedder,
            vision_embedder=vision_embedder,
            force=force,
        )
        batch.results.append(result)

    return batch
