from __future__ import annotations

from pathlib import Path

import typer

from video_lance import discovery, pipeline, search, store
from video_lance.config import (
    DEFAULT_INCLUDE,
    Config,
    FrameSamplingConfig,
    SegmentationConfig,
)
from video_lance.embed_text import DEFAULT_TEXT_MODEL, get_text_embedder
from video_lance.embed_vision import DEFAULT_VISION_MODEL, get_vision_embedder
from video_lance.transcribe import get_transcriber

app = typer.Typer(
    add_completion=False,
    help="video-lance: index a directory of videos into a LanceDB store.",
    no_args_is_help=True,
)


@app.callback()
def _root() -> None:
    """video-lance: pipelines and search over a LanceDB video index."""
    # The presence of this callback forces typer to keep `ingest` (and future
    # `search`/`info`/`reindex`) as subcommands rather than collapsing them.


def _split_csv(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


@app.command()
def ingest(
    path: Path = typer.Argument(..., exists=True, help="Directory to walk, or a single video."),
    segment_seconds: float = typer.Option(30.0, "--segment-seconds", "-s"),
    overlap_seconds: float = typer.Option(0.0, "--overlap-seconds", "-o"),
    merge_short_tail: bool = typer.Option(True, "--merge-short-tail/--no-merge-short-tail"),
    min_tail_seconds: float = typer.Option(
        5.0,
        "--min-tail-seconds",
        help="Merge a trailing segment shorter than this into the prior one.",
    ),
    sentence_snap_tolerance: float = typer.Option(0.0, "--sentence-snap-tolerance"),
    frame_position: float = typer.Option(0.5, "--frame-position"),
    frame_jpeg_quality: int = typer.Option(85, "--frame-jpeg-quality"),
    frame_max_long_edge: int = typer.Option(512, "--frame-max-long-edge"),
    workers: int = typer.Option(1, "--workers", min=1),
    device: str = typer.Option("auto", "--device", help="auto|cuda|mps|cpu"),
    whisper_model: str = typer.Option("small.en", "--whisper-model"),
    text_embed_model: str = typer.Option(
        "intfloat/multilingual-e5-large-instruct", "--text-embed-model"
    ),
    vision_embed_model: str = typer.Option(
        "google/siglip2-so400m-patch14-384", "--vision-embed-model"
    ),
    db_path: Path = typer.Option(Path("./video-lance.db"), "--db-path"),
    include: str = typer.Option(",".join(DEFAULT_INCLUDE), "--include"),
    exclude: str = typer.Option("", "--exclude"),
    force: bool = typer.Option(False, "--force", help="Re-ingest even if already indexed."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Walk + probe + list files; no writes."),
) -> None:
    """Index a directory (or a single video) into the LanceDB store."""
    cfg = Config(
        segmentation=SegmentationConfig(
            segment_seconds=segment_seconds,
            overlap_seconds=overlap_seconds,
            merge_short_tail=merge_short_tail,
            min_tail_seconds=min_tail_seconds,
            sentence_snap_tolerance_seconds=sentence_snap_tolerance,
        ),
        frames=FrameSamplingConfig(
            position=frame_position,
            jpeg_quality=frame_jpeg_quality,
            max_long_edge=frame_max_long_edge,
        ),
        whisper_model=whisper_model,
        text_embed_model=text_embed_model,
        vision_embed_model=vision_embed_model,
        device=device,
        db_path=db_path,
        include=_split_csv(include) or DEFAULT_INCLUDE,
        exclude=_split_csv(exclude),
        workers=workers,
    )

    paths = discovery.walk(path, include=cfg.include, exclude=cfg.exclude)
    if not paths:
        typer.echo(f"no videos matched under {path}")
        raise typer.Exit(code=1)

    typer.echo(f"discovered {len(paths)} video(s) under {path}")

    if dry_run:
        for p in paths:
            typer.echo(f"  {p}")
        return

    db = store.connect(cfg.db_path)
    tables = store.ensure_tables(db)
    store.set_embedding_models(
        tables,
        text_embed_model=cfg.text_embed_model,
        vision_embed_model=cfg.vision_embed_model,
    )

    typer.echo("loading transcriber model")
    transcriber = get_transcriber(cfg.whisper_model, device=cfg.device)
    typer.echo("loading text embedding model")
    text_embedder = get_text_embedder(cfg.text_embed_model, device=cfg.device)
    typer.echo("loading vision embedding model")
    vision_embedder = get_vision_embedder(cfg.vision_embed_model, device=cfg.device)

    batch = pipeline.process_directory(
        path,
        cfg,
        tables,
        transcriber=transcriber,
        text_embedder=text_embedder,
        vision_embedder=vision_embedder,
        force=force,
    )

    for r in batch.results:
        if r.ok and not r.skipped:
            typer.echo(f"  ok      {r.path}  ({r.segments_written} segments)")
        elif r.skipped:
            typer.echo(f"  skip    {r.path}  ({r.skip_reason or 'already indexed'})")
        else:
            typer.echo(f"  FAILED  {r.path}  {r.error}")

    typer.echo(
        f"done — succeeded={batch.succeeded} skipped={batch.skipped} "
        f"failed={batch.failed} segments_written={batch.written}"
    )
    if batch.failed:
        raise typer.Exit(code=2)


@app.command("search")
def search_cmd(
    query: str = typer.Argument("", help="Text query (omit if using --image)."),
    image: Path | None = typer.Option(None, "--image", help="Use an image as query (visual mode)."),
    limit: int = typer.Option(10, "--limit", min=1),
    mode: str = typer.Option("text", "--mode", help="text | visual | multi"),
    visual_weight: float = typer.Option(0.4, "--visual-weight", min=0.0, max=1.0),
    sql_filter: str | None = typer.Option(None, "--filter", help="SQL WHERE expression"),
    db_path: Path = typer.Option(Path("./video-lance.db"), "--db-path"),
    device: str = typer.Option("auto", "--device"),
    text_embed_model: str | None = typer.Option(None, "--text-embed-model"),
    vision_embed_model: str | None = typer.Option(None, "--vision-embed-model"),
) -> None:
    """Search the LanceDB store in text / visual / multi mode."""
    if mode not in {"text", "visual", "multi"}:
        typer.echo(f"unknown --mode {mode!r}; expected text | visual | multi", err=True)
        raise typer.Exit(code=2)
    if not db_path.exists():
        typer.echo(f"no database at {db_path}", err=True)
        raise typer.Exit(code=2)

    db = store.connect(db_path)
    tables = store.ensure_tables(db)

    # Default models come from what the DB was built with; fall back to built-in defaults.
    stored_text, stored_vision = store.get_embedding_models(tables)
    text_model = text_embed_model or stored_text or DEFAULT_TEXT_MODEL
    vision_model = vision_embed_model or stored_vision or DEFAULT_VISION_MODEL

    if mode == "text":
        if not query:
            typer.echo("text mode requires a QUERY string", err=True)
            raise typer.Exit(code=2)
        text_embedder = get_text_embedder(text_model, device=device)
        hits = search.search_text(tables, text_embedder, query, limit=limit, sql_filter=sql_filter)
    elif mode == "visual":
        if image is None and not query:
            typer.echo("visual mode requires QUERY or --image", err=True)
            raise typer.Exit(code=2)
        vision_embedder = get_vision_embedder(vision_model, device=device)
        hits = search.search_visual(
            tables,
            vision_embedder,
            query=query if image is None else None,
            image=image,
            limit=limit,
            sql_filter=sql_filter,
        )
    else:  # multi
        if not query:
            typer.echo("multi mode requires a QUERY string", err=True)
            raise typer.Exit(code=2)
        text_embedder = get_text_embedder(text_model, device=device)
        vision_embedder = get_vision_embedder(vision_model, device=device)
        hits = search.search_multi(
            tables,
            text_embedder,
            vision_embedder,
            query,
            limit=limit,
            visual_weight=visual_weight,
            sql_filter=sql_filter,
        )

    if not hits:
        typer.echo("no results")
        return

    for rank, hit in enumerate(hits, start=1):
        snippet = (hit.text or "").strip().replace("\n", " ")
        if len(snippet) > 120:
            snippet = snippet[:117] + "..."
        path = hit.relative_path or hit.source_path
        typer.echo(f"{rank}. [{hit.score:.3f}] {path} @ {hit.time_range()}")
        if snippet:
            typer.echo(f'   "{snippet}"')
        typer.echo(f"   open: {hit.deep_link()}")


@app.command("info")
def info_cmd(
    db_path: Path = typer.Option(Path("./video-lance.db"), "--db-path"),
) -> None:
    """Report row counts, embedding models, and segment-table indexes."""
    if not db_path.exists():
        typer.echo(f"no database at {db_path}", err=True)
        raise typer.Exit(code=2)
    db = store.connect(db_path)
    tables = store.ensure_tables(db)
    info = search.db_info(tables, db_path=db_path)

    typer.echo(f"db: {info.db_path}")
    typer.echo(f"videos:   {info.videos}")
    typer.echo(f"segments: {info.segments}")
    typer.echo("embedding models:")
    typer.echo(f"  text:   {info.text_embed_model or '(not set)'}")
    typer.echo(f"  vision: {info.vision_embed_model or '(not set)'}")
    typer.echo("segment indexes:")
    if info.segment_indexes:
        for name in info.segment_indexes:
            typer.echo(f"  - {name}")
    else:
        typer.echo("  (none)")


@app.command("reindex")
def reindex_cmd(
    db_path: Path = typer.Option(Path("./video-lance.db"), "--db-path"),
) -> None:
    """Drop and rebuild the FTS + vector indexes on the `segments` table."""
    if not db_path.exists():
        typer.echo(f"no database at {db_path}", err=True)
        raise typer.Exit(code=2)
    db = store.connect(db_path)
    tables = store.ensure_tables(db)
    status = search.ensure_indexes(tables, replace=True)
    typer.echo(f"fts_text:           {status.fts_text}")
    typer.echo(f"vec_text_embedding: {status.vec_text}")
    typer.echo(f"vec_visual_embed:   {status.vec_visual}")


@app.command("ui")
def ui_cmd(
    db_path: Path = typer.Option(Path("./video-lance.db"), "--db-path"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(7860, "--port"),
    device: str = typer.Option("auto", "--device"),
    share: bool = typer.Option(False, "--share", help="Ask Gradio for a temporary public URL."),
) -> None:
    """Launch the Gradio web UI.

    Loads the LanceDB store at `--db-path`, lazily instantiates the e5 +
    SigLIP embedders (so queries are encoded on the server, not in the
    browser), and starts a Gradio server on `--host:--port`.
    """
    # Lazy import — gradio brings in a sizeable web stack and we don't want
    # `video-lance --help` to pay for it.
    from video_lance.ui_app import launch

    launch(db_path=db_path, host=host, port=port, device=device, share=share)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
