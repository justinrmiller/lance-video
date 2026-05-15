from __future__ import annotations

import pyarrow as pa

# Canonical dimension constants. Imported wherever they're needed — never
# hardcoded in two places. The embedder modules re-export these.
TEXT_EMBED_DIM = 1024  # intfloat/multilingual-e5-large-instruct
VISION_EMBED_DIM = 1152  # google/siglip2-so400m-patch14-384 (same dim as SigLIP 1)

VIDEOS_TABLE = "videos"
SEGMENTS_TABLE = "segments"
METADATA_TABLE = "_metadata"

# Blob V2 opt-in: Lance picks Inline / Packed / Dedicated storage automatically
# based on payload size. Tagging is the only thing the schema author has to do.
_BLOB_METADATA = {"lance-encoding:blob": "true"}


videos_schema = pa.schema(
    [
        pa.field("video_id", pa.string()),  # sha256(absolute_path)[:16]
        pa.field("source_path", pa.string()),
        pa.field("relative_path", pa.string()),
        pa.field("duration_s", pa.float32()),
        pa.field("fps", pa.float32()),
        pa.field("width", pa.int32()),
        pa.field("height", pa.int32()),
        pa.field("codec", pa.string()),
        pa.field("size_bytes", pa.int64()),
        pa.field("ingested_at", pa.timestamp("us")),
        pa.field("segment_seconds", pa.float32()),
        pa.field("overlap_seconds", pa.float32()),
        pa.field("transcript_full", pa.string()),
    ]
)


segments_schema = pa.schema(
    [
        pa.field("segment_id", pa.string()),  # f"{video_id}:{idx:06d}"
        pa.field("video_id", pa.string()),
        pa.field("idx", pa.int32()),
        pa.field("start_s", pa.float32()),
        pa.field("end_s", pa.float32()),
        pa.field("keyframe_t_s", pa.float32()),
        pa.field("text", pa.string()),
        pa.field("text_embedding", pa.list_(pa.float32(), TEXT_EMBED_DIM)),
        pa.field("visual_embedding", pa.list_(pa.float32(), VISION_EMBED_DIM)),
        pa.field("clip_bytes", pa.large_binary(), metadata=_BLOB_METADATA),
        pa.field("keyframe_jpeg", pa.large_binary(), metadata=_BLOB_METADATA),
    ]
)


# Small free-form key/value table used to persist things like the embedding
# model identifiers the DB was built with. Per PLAN §10 Session 4, this is
# what `info` reads and what re-ingest checks against to detect a model swap.
metadata_schema = pa.schema(
    [
        pa.field("key", pa.string()),
        pa.field("value", pa.string()),
    ]
)
