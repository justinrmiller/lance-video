from __future__ import annotations

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


def test_text_embed_dim_canonical() -> None:
    # The constant must match the e5-instruct model's actual output dim.
    assert TEXT_EMBED_DIM == 1024


def test_vision_embed_dim_canonical() -> None:
    assert VISION_EMBED_DIM == 1152


def test_videos_schema_fields() -> None:
    expected = {
        "video_id": pa.string(),
        "source_path": pa.string(),
        "relative_path": pa.string(),
        "duration_s": pa.float32(),
        "fps": pa.float32(),
        "width": pa.int32(),
        "height": pa.int32(),
        "codec": pa.string(),
        "size_bytes": pa.int64(),
        "segment_seconds": pa.float32(),
        "overlap_seconds": pa.float32(),
        "transcript_full": pa.string(),
    }
    for name, dtype in expected.items():
        field = videos_schema.field(name)
        assert field.type == dtype, f"{name}: {field.type} != {dtype}"
    # Timestamp field is `us` resolution.
    assert videos_schema.field("ingested_at").type == pa.timestamp("us")


def test_segments_schema_fields() -> None:
    text_emb = segments_schema.field("text_embedding")
    assert text_emb.type == pa.list_(pa.float32(), TEXT_EMBED_DIM)
    vis_emb = segments_schema.field("visual_embedding")
    assert vis_emb.type == pa.list_(pa.float32(), VISION_EMBED_DIM)
    assert segments_schema.field("segment_id").type == pa.string()
    assert segments_schema.field("video_id").type == pa.string()
    assert segments_schema.field("idx").type == pa.int32()


def test_blob_metadata_tags_present() -> None:
    """Blob V1 opt-in is the whole point of this project — the tag MUST be on
    both blob columns or LanceDB will store them inline as regular bytes."""
    for column in ("clip_bytes", "keyframe_jpeg"):
        field = segments_schema.field(column)
        assert field.type == pa.large_binary(), column
        meta = field.metadata or {}
        # Arrow metadata keys/values are bytes.
        assert meta.get(b"lance-encoding:blob") == b"true", (
            f"{column} missing Blob V1 metadata tag (got {meta})"
        )


def test_metadata_schema_is_kv() -> None:
    assert metadata_schema.field("key").type == pa.string()
    assert metadata_schema.field("value").type == pa.string()


def test_table_name_constants() -> None:
    assert VIDEOS_TABLE == "videos"
    assert SEGMENTS_TABLE == "segments"
    assert METADATA_TABLE == "_metadata"
