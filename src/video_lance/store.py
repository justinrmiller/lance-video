from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa

from video_lance.schema import (
    METADATA_TABLE,
    SEGMENTS_TABLE,
    TEXT_EMBED_DIM,
    VIDEOS_TABLE,
    VISION_EMBED_DIM,
    metadata_schema,
    segments_schema,
    videos_schema,
)

# -- ids ----------------------------------------------------------------------


def video_id_for_path(path: Path) -> str:
    """Stable identifier for a video file: sha256 of the absolute path, first
    16 hex chars.

    Note: moving the source file invalidates this ID. Acceptable for v1;
    content-hash v2 is a known follow-up.
    """
    absolute = str(Path(path).resolve())
    return hashlib.sha256(absolute.encode("utf-8")).hexdigest()[:16]


def segment_id_for(video_id: str, idx: int) -> str:
    return f"{video_id}:{idx:06d}"


# -- records ------------------------------------------------------------------


@dataclass
class VideoRow:
    """Wire-level row for the `videos` table."""

    video_id: str
    source_path: str
    relative_path: str
    duration_s: float
    fps: float
    width: int
    height: int
    codec: str
    size_bytes: int
    ingested_at: datetime
    segment_seconds: float
    overlap_seconds: float
    transcript_full: str

    def to_pyarrow_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "source_path": self.source_path,
            "relative_path": self.relative_path,
            "duration_s": float(self.duration_s),
            "fps": float(self.fps),
            "width": int(self.width),
            "height": int(self.height),
            "codec": self.codec,
            "size_bytes": int(self.size_bytes),
            "ingested_at": self.ingested_at,
            "segment_seconds": float(self.segment_seconds),
            "overlap_seconds": float(self.overlap_seconds),
            "transcript_full": self.transcript_full,
        }


@dataclass
class SegmentRow:
    """Wire-level row for the `segments` table."""

    segment_id: str
    video_id: str
    idx: int
    start_s: float
    end_s: float
    keyframe_t_s: float
    text: str
    text_embedding: Sequence[float]
    visual_embedding: Sequence[float]
    clip_bytes: bytes
    keyframe_jpeg: bytes

    def __post_init__(self) -> None:
        if len(self.text_embedding) != TEXT_EMBED_DIM:
            raise ValueError(
                f"text_embedding has dim {len(self.text_embedding)}, expected {TEXT_EMBED_DIM}"
            )
        if len(self.visual_embedding) != VISION_EMBED_DIM:
            raise ValueError(
                f"visual_embedding has dim {len(self.visual_embedding)}, "
                f"expected {VISION_EMBED_DIM}"
            )

    def to_pyarrow_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "video_id": self.video_id,
            "idx": int(self.idx),
            "start_s": float(self.start_s),
            "end_s": float(self.end_s),
            "keyframe_t_s": float(self.keyframe_t_s),
            "text": self.text,
            "text_embedding": [float(x) for x in self.text_embedding],
            "visual_embedding": [float(x) for x in self.visual_embedding],
            "clip_bytes": bytes(self.clip_bytes),
            "keyframe_jpeg": bytes(self.keyframe_jpeg),
        }


# -- tables -------------------------------------------------------------------


@dataclass
class StoreTables:
    db: lancedb.DBConnection
    videos: lancedb.table.Table
    segments: lancedb.table.Table
    metadata: lancedb.table.Table


def connect(db_path: Path | str) -> lancedb.DBConnection:
    return lancedb.connect(str(db_path))


def _list_table_names(db: lancedb.DBConnection) -> list[str]:
    """Return the names of tables in `db`.

    `DBConnection.list_tables()` returns a paginated response object in newer
    lancedb versions; older versions returned a plain list. This helper papers
    over the difference. The `str(...)` coercion both flattens that union for
    the type checkers and guards against unexpected shapes.
    """
    result = db.list_tables()
    tables = getattr(result, "tables", result)
    return [str(name) for name in tables]


def _open_or_create(db: lancedb.DBConnection, name: str, schema: pa.Schema) -> lancedb.table.Table:
    if name in _list_table_names(db):
        return db.open_table(name)
    return db.create_table(name, schema=schema)


def ensure_tables(db: lancedb.DBConnection) -> StoreTables:
    """Open or create all three tables. Idempotent: existing data is preserved.

    Re-running this on an existing DB just opens the tables in place — it never
    rewrites or migrates schemas.
    """
    videos = _open_or_create(db, VIDEOS_TABLE, videos_schema)
    segments = _open_or_create(db, SEGMENTS_TABLE, segments_schema)
    metadata = _open_or_create(db, METADATA_TABLE, metadata_schema)
    return StoreTables(db=db, videos=videos, segments=segments, metadata=metadata)


# -- video upsert -------------------------------------------------------------


def _escape_sql_str(value: str) -> str:
    return value.replace("'", "''")


def get_video(tables: StoreTables, video_id: str) -> dict[str, Any] | None:
    rows = (
        tables.videos.search()
        .where(f"video_id = '{_escape_sql_str(video_id)}'")
        .limit(1)
        .to_arrow()
        .to_pylist()
    )
    return rows[0] if rows else None


def upsert_video(tables: StoreTables, row: VideoRow) -> None:
    """Replace any existing `videos` row with the same video_id and add `row`."""
    tables.videos.delete(f"video_id = '{_escape_sql_str(row.video_id)}'")
    tables.videos.add([row.to_pyarrow_dict()])


def delete_video(tables: StoreTables, video_id: str) -> None:
    """Remove the video row and all its segments."""
    escaped = _escape_sql_str(video_id)
    tables.videos.delete(f"video_id = '{escaped}'")
    tables.segments.delete(f"video_id = '{escaped}'")


# -- segment upsert -----------------------------------------------------------


def upsert_segments(tables: StoreTables, rows: Iterable[SegmentRow]) -> int:
    """Replace all existing segments for the video_ids appearing in `rows`
    with the given rows. Returns the number of rows written.

    The delete-then-add is scoped per-video-id rather than per-segment-id so
    that re-segmenting a video with a different segment length cleans up the
    stale rows in one shot.
    """
    rows = list(rows)
    if not rows:
        return 0

    video_ids = sorted({r.video_id for r in rows})
    quoted = ", ".join(f"'{_escape_sql_str(vid)}'" for vid in video_ids)
    tables.segments.delete(f"video_id IN ({quoted})")

    payload = [r.to_pyarrow_dict() for r in rows]
    tables.segments.add(payload)
    return len(payload)


def count_segments_for_video(tables: StoreTables, video_id: str) -> int:
    """Number of segments stored for `video_id` (no row payload transferred)."""
    return int(tables.segments.count_rows(f"video_id = '{_escape_sql_str(video_id)}'"))


def get_segments_for_video(tables: StoreTables, video_id: str) -> list[dict[str, Any]]:
    where = f"video_id = '{_escape_sql_str(video_id)}'"
    n = int(tables.segments.count_rows(where))
    if n == 0:
        return []
    rows: list[dict[str, Any]] = (
        tables.segments.search().where(where).limit(n).to_arrow().to_pylist()
    )
    return rows


def get_videos_by_ids(tables: StoreTables, video_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
    """Fetch just the `videos` rows for `video_ids`, keyed by video_id.

    Used to resolve search hits back to their source video without loading the
    whole `videos` table on every query.
    """
    ids = sorted({vid for vid in video_ids})
    if not ids:
        return {}
    quoted = ", ".join(f"'{_escape_sql_str(vid)}'" for vid in ids)
    rows: list[dict[str, Any]] = (
        tables.videos.search()
        .where(f"video_id IN ({quoted})")
        .limit(len(ids))
        .to_arrow()
        .to_pylist()
    )
    return {str(r["video_id"]): r for r in rows}


# -- blob reads ---------------------------------------------------------------

# Blob V1 columns (`clip_bytes`, `keyframe_jpeg`) round-trip through standard
# reads as descriptors, not raw bytes. To get bytes back we resolve the
# segment to its row id, then ask the underlying lance dataset for the blob.

BLOB_COLUMNS = frozenset({"clip_bytes", "keyframe_jpeg"})


def read_segment_blob(tables: StoreTables, segment_id: str, column: str) -> bytes:
    """Fetch the raw bytes from a blob column for a single segment."""
    if column not in BLOB_COLUMNS:
        raise ValueError(f"{column!r} is not a blob column; expected one of {sorted(BLOB_COLUMNS)}")

    arrow = (
        tables.segments.search()
        .where(f"segment_id = '{_escape_sql_str(segment_id)}'")
        .with_row_id(True)
        .limit(1)
        .to_arrow()
    )
    row_ids = arrow.column("_rowid").to_pylist()
    if not row_ids:
        raise KeyError(f"segment {segment_id!r} not found")

    ds = tables.segments.to_lance()
    blob_files = ds.take_blobs(column, ids=[row_ids[0]])
    return bytes(blob_files[0].read())


# -- metadata -----------------------------------------------------------------


def get_metadata(tables: StoreTables) -> dict[str, str]:
    rows = tables.metadata.to_arrow().to_pylist()
    return {str(r["key"]): str(r["value"]) for r in rows}


def set_metadata(tables: StoreTables, items: dict[str, str]) -> None:
    """Upsert each key/value into the metadata table."""
    if not items:
        return
    keys = sorted(items.keys())
    quoted = ", ".join(f"'{_escape_sql_str(k)}'" for k in keys)
    tables.metadata.delete(f"key IN ({quoted})")
    tables.metadata.add([{"key": k, "value": str(v)} for k, v in items.items()])


# -- model identifiers (a convenience layer over `metadata`) ------------------

KEY_TEXT_MODEL = "text_embed_model"
KEY_VISION_MODEL = "vision_embed_model"


def set_embedding_models(
    tables: StoreTables,
    *,
    text_embed_model: str,
    vision_embed_model: str,
) -> None:
    """Persist the embedding model identifiers the DB was built with.

    Called once on first ingest. Re-ingest with different model identifiers
    is a higher-level concern handled by the caller.
    """
    set_metadata(
        tables,
        {KEY_TEXT_MODEL: text_embed_model, KEY_VISION_MODEL: vision_embed_model},
    )


def get_embedding_models(tables: StoreTables) -> tuple[str | None, str | None]:
    md = get_metadata(tables)
    return md.get(KEY_TEXT_MODEL), md.get(KEY_VISION_MODEL)


class EmbeddingModelMismatch(RuntimeError):
    """Raised when an ingest requests embedding models that differ from the
    ones the store was originally built with."""


def assert_or_set_embedding_models(
    tables: StoreTables,
    *,
    text_embed_model: str,
    vision_embed_model: str,
    force: bool = False,
) -> None:
    """Record the embedding model identifiers, or reject a mismatch.

    On a fresh store (no identifiers persisted yet) this records the given
    models. On a store that already has identifiers, it raises
    `EmbeddingModelMismatch` if either requested model differs from the stored
    one — writing vectors from a different model into the same fixed-dim
    column would silently corrupt search, since the dimensions can still match.

    `force=True` overrides the check and overwrites the stored identifiers (the
    caller is then responsible for having re-embedded every existing segment,
    e.g. via a full `--force` re-ingest of the whole store).
    """
    stored_text, stored_vision = get_embedding_models(tables)

    if not force and stored_text is not None and stored_text != text_embed_model:
        raise EmbeddingModelMismatch(
            f"store was built with text embedding model {stored_text!r}, "
            f"refusing to ingest with {text_embed_model!r}; re-ingest the whole "
            f"store with --force to switch models"
        )
    if not force and stored_vision is not None and stored_vision != vision_embed_model:
        raise EmbeddingModelMismatch(
            f"store was built with vision embedding model {stored_vision!r}, "
            f"refusing to ingest with {vision_embed_model!r}; re-ingest the whole "
            f"store with --force to switch models"
        )

    set_embedding_models(
        tables,
        text_embed_model=text_embed_model,
        vision_embed_model=vision_embed_model,
    )
