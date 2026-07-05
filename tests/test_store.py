from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from video_lance.schema import (
    METADATA_TABLE,
    SEGMENTS_TABLE,
    TEXT_EMBED_DIM,
    VIDEOS_TABLE,
    VISION_EMBED_DIM,
)
from video_lance.store import (
    EmbeddingModelMismatch,
    SegmentRow,
    VideoRow,
    _list_table_names,
    assert_or_set_embedding_models,
    connect,
    count_segments_for_video,
    delete_video,
    ensure_tables,
    get_embedding_models,
    get_metadata,
    get_segments_for_video,
    get_video,
    get_videos_by_ids,
    read_segment_blob,
    segment_id_for,
    set_embedding_models,
    set_metadata,
    upsert_segments,
    upsert_video,
    video_id_for_path,
)

# -- ids ---------------------------------------------------------------------


def test_video_id_stable_for_same_path(tmp_path: Path) -> None:
    p = tmp_path / "a.mp4"
    p.write_bytes(b"x")
    assert video_id_for_path(p) == video_id_for_path(p)


def test_video_id_differs_for_different_paths(tmp_path: Path) -> None:
    p1 = tmp_path / "a.mp4"
    p2 = tmp_path / "b.mp4"
    p1.write_bytes(b"x")
    p2.write_bytes(b"x")
    assert video_id_for_path(p1) != video_id_for_path(p2)


def test_video_id_length() -> None:
    vid = video_id_for_path(Path("/tmp/whatever.mp4"))
    assert len(vid) == 16
    int(vid, 16)  # must be valid hex


def test_segment_id_format() -> None:
    assert segment_id_for("abc123", 7) == "abc123:000007"


# -- helpers -----------------------------------------------------------------


def _random_vec(dim: int, rng: np.random.Generator) -> list[float]:
    return rng.standard_normal(dim).astype(np.float32).tolist()


def _make_video_row(video_id: str, *, segment_seconds: float = 2.0) -> VideoRow:
    return VideoRow(
        video_id=video_id,
        source_path=f"/abs/{video_id}.mp4",
        relative_path=f"{video_id}.mp4",
        duration_s=10.0,
        fps=30.0,
        width=320,
        height=240,
        codec="h264",
        size_bytes=12345,
        ingested_at=datetime.now(UTC),
        segment_seconds=segment_seconds,
        overlap_seconds=0.0,
        transcript_full="hello world",
    )


def _make_segment_row(
    video_id: str, idx: int, rng: np.random.Generator, *, clip_payload: bytes | None = None
) -> SegmentRow:
    payload = clip_payload if clip_payload is not None else f"clip-{video_id}-{idx}".encode() * 50
    return SegmentRow(
        segment_id=segment_id_for(video_id, idx),
        video_id=video_id,
        idx=idx,
        start_s=float(idx * 2),
        end_s=float((idx + 1) * 2),
        keyframe_t_s=float(idx * 2 + 1),
        text=f"text for segment {idx}",
        text_embedding=_random_vec(TEXT_EMBED_DIM, rng),
        visual_embedding=_random_vec(VISION_EMBED_DIM, rng),
        clip_bytes=payload,
        keyframe_jpeg=f"jpeg-{video_id}-{idx}".encode(),
    )


# -- ensure_tables -----------------------------------------------------------


def test_ensure_tables_creates_all_three(tmp_path: Path) -> None:
    db = connect(tmp_path / "db")
    ensure_tables(db)
    names = set(_list_table_names(db))
    assert {VIDEOS_TABLE, SEGMENTS_TABLE, METADATA_TABLE}.issubset(names)


def test_ensure_tables_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "db"
    db = connect(db_path)
    t1 = ensure_tables(db)
    rng = np.random.default_rng(0)
    vid = "0123456789abcdef"
    upsert_video(t1, _make_video_row(vid))
    upsert_segments(t1, [_make_segment_row(vid, 0, rng)])

    # Re-open and re-ensure: the data must still be there.
    db2 = connect(db_path)
    t2 = ensure_tables(db2)
    assert t2.videos.count_rows() == 1
    assert t2.segments.count_rows() == 1


# -- video upsert ------------------------------------------------------------


def test_upsert_video_replaces_existing(tmp_path: Path) -> None:
    tables = ensure_tables(connect(tmp_path / "db"))
    vid = "deadbeefdeadbeef"
    upsert_video(tables, _make_video_row(vid, segment_seconds=2.0))
    upsert_video(tables, _make_video_row(vid, segment_seconds=5.0))
    assert tables.videos.count_rows() == 1
    row = get_video(tables, vid)
    assert row is not None
    assert row["segment_seconds"] == pytest.approx(5.0)


def test_get_video_returns_none_when_missing(tmp_path: Path) -> None:
    tables = ensure_tables(connect(tmp_path / "db"))
    assert get_video(tables, "missingmissing") is None


# -- segments upsert ---------------------------------------------------------


def test_segment_roundtrip_preserves_vectors_and_blobs(tmp_path: Path) -> None:
    tables = ensure_tables(connect(tmp_path / "db"))
    rng = np.random.default_rng(42)
    vid = "abcdef0123456789"
    payload = b"a deterministic clip payload " * 32
    row = _make_segment_row(vid, 0, rng, clip_payload=payload)
    expected_text = list(row.text_embedding)
    expected_visual = list(row.visual_embedding)
    expected_jpeg = row.keyframe_jpeg

    upsert_segments(tables, [row])
    seg_rows = get_segments_for_video(tables, vid)
    assert len(seg_rows) == 1
    seg = seg_rows[0]

    assert seg["segment_id"] == row.segment_id
    assert seg["text_embedding"] == pytest.approx(expected_text, abs=1e-6)
    assert seg["visual_embedding"] == pytest.approx(expected_visual, abs=1e-6)

    # Blob columns come back as descriptors via standard reads; resolve them.
    assert read_segment_blob(tables, row.segment_id, "clip_bytes") == payload
    assert read_segment_blob(tables, row.segment_id, "keyframe_jpeg") == expected_jpeg


def test_upsert_segments_replaces_by_video_id(tmp_path: Path) -> None:
    tables = ensure_tables(connect(tmp_path / "db"))
    rng = np.random.default_rng(1)
    vid = "1111222233334444"

    upsert_segments(tables, [_make_segment_row(vid, i, rng) for i in range(3)])
    assert tables.segments.count_rows() == 3

    # Re-ingesting with two new segments must replace the old three.
    upsert_segments(tables, [_make_segment_row(vid, i, rng) for i in range(2)])
    rows = get_segments_for_video(tables, vid)
    assert len(rows) == 2
    assert {r["idx"] for r in rows} == {0, 1}


def test_upsert_segments_does_not_touch_other_videos(tmp_path: Path) -> None:
    tables = ensure_tables(connect(tmp_path / "db"))
    rng = np.random.default_rng(2)
    a, b = "aaaa1111aaaa1111", "bbbb2222bbbb2222"
    upsert_segments(tables, [_make_segment_row(a, i, rng) for i in range(2)])
    upsert_segments(tables, [_make_segment_row(b, i, rng) for i in range(3)])
    assert tables.segments.count_rows() == 5
    # Re-upsert just A — B's rows must survive untouched.
    upsert_segments(tables, [_make_segment_row(a, 0, rng)])
    assert len(get_segments_for_video(tables, a)) == 1
    assert len(get_segments_for_video(tables, b)) == 3


def test_empty_upsert_is_noop(tmp_path: Path) -> None:
    tables = ensure_tables(connect(tmp_path / "db"))
    rng = np.random.default_rng(3)
    upsert_segments(tables, [_make_segment_row("xxxx", 0, rng)])
    n = upsert_segments(tables, [])
    assert n == 0
    assert tables.segments.count_rows() == 1


def test_segment_row_rejects_wrong_text_dim() -> None:
    with pytest.raises(ValueError, match="text_embedding"):
        SegmentRow(
            segment_id="x:000000",
            video_id="x",
            idx=0,
            start_s=0.0,
            end_s=1.0,
            keyframe_t_s=0.5,
            text="",
            text_embedding=[0.0] * (TEXT_EMBED_DIM - 1),
            visual_embedding=[0.0] * VISION_EMBED_DIM,
            clip_bytes=b"",
            keyframe_jpeg=b"",
        )


def test_segment_row_rejects_wrong_visual_dim() -> None:
    with pytest.raises(ValueError, match="visual_embedding"):
        SegmentRow(
            segment_id="x:000000",
            video_id="x",
            idx=0,
            start_s=0.0,
            end_s=1.0,
            keyframe_t_s=0.5,
            text="",
            text_embedding=[0.0] * TEXT_EMBED_DIM,
            visual_embedding=[0.0] * (VISION_EMBED_DIM + 1),
            clip_bytes=b"",
            keyframe_jpeg=b"",
        )


# -- delete ------------------------------------------------------------------


def test_delete_video_removes_segments_too(tmp_path: Path) -> None:
    tables = ensure_tables(connect(tmp_path / "db"))
    rng = np.random.default_rng(4)
    vid = "9999888877776666"
    upsert_video(tables, _make_video_row(vid))
    upsert_segments(tables, [_make_segment_row(vid, i, rng) for i in range(3)])
    assert tables.videos.count_rows() == 1
    assert tables.segments.count_rows() == 3

    delete_video(tables, vid)
    assert tables.videos.count_rows() == 0
    assert tables.segments.count_rows() == 0


# -- metadata ----------------------------------------------------------------


def test_metadata_set_and_get(tmp_path: Path) -> None:
    tables = ensure_tables(connect(tmp_path / "db"))
    set_metadata(tables, {"a": "1", "b": "two"})
    assert get_metadata(tables) == {"a": "1", "b": "two"}


def test_metadata_set_is_upsert(tmp_path: Path) -> None:
    tables = ensure_tables(connect(tmp_path / "db"))
    set_metadata(tables, {"text_embed_model": "v1"})
    set_metadata(tables, {"text_embed_model": "v2", "extra": "hello"})
    md = get_metadata(tables)
    assert md["text_embed_model"] == "v2"
    assert md["extra"] == "hello"


def test_embedding_models_helpers(tmp_path: Path) -> None:
    tables = ensure_tables(connect(tmp_path / "db"))
    assert get_embedding_models(tables) == (None, None)
    set_embedding_models(tables, text_embed_model="intfloat/foo", vision_embed_model="google/bar")
    assert get_embedding_models(tables) == ("intfloat/foo", "google/bar")


# -- embedding-model guard ---------------------------------------------------


def test_assert_or_set_records_on_fresh_store(tmp_path: Path) -> None:
    tables = ensure_tables(connect(tmp_path / "db"))
    assert_or_set_embedding_models(tables, text_embed_model="text-a", vision_embed_model="vis-a")
    assert get_embedding_models(tables) == ("text-a", "vis-a")


def test_assert_or_set_allows_matching_models(tmp_path: Path) -> None:
    tables = ensure_tables(connect(tmp_path / "db"))
    set_embedding_models(tables, text_embed_model="text-a", vision_embed_model="vis-a")
    # Same ids: no error, no change.
    assert_or_set_embedding_models(tables, text_embed_model="text-a", vision_embed_model="vis-a")
    assert get_embedding_models(tables) == ("text-a", "vis-a")


def test_assert_or_set_rejects_text_mismatch(tmp_path: Path) -> None:
    tables = ensure_tables(connect(tmp_path / "db"))
    set_embedding_models(tables, text_embed_model="text-a", vision_embed_model="vis-a")
    with pytest.raises(EmbeddingModelMismatch, match="text"):
        assert_or_set_embedding_models(
            tables, text_embed_model="text-b", vision_embed_model="vis-a"
        )
    # Stored ids must be left untouched by the rejected call.
    assert get_embedding_models(tables) == ("text-a", "vis-a")


def test_assert_or_set_rejects_vision_mismatch(tmp_path: Path) -> None:
    tables = ensure_tables(connect(tmp_path / "db"))
    set_embedding_models(tables, text_embed_model="text-a", vision_embed_model="vis-a")
    with pytest.raises(EmbeddingModelMismatch, match="vision"):
        assert_or_set_embedding_models(
            tables, text_embed_model="text-a", vision_embed_model="vis-b"
        )


def test_assert_or_set_force_overwrites(tmp_path: Path) -> None:
    tables = ensure_tables(connect(tmp_path / "db"))
    set_embedding_models(tables, text_embed_model="text-a", vision_embed_model="vis-a")
    assert_or_set_embedding_models(
        tables, text_embed_model="text-b", vision_embed_model="vis-b", force=True
    )
    assert get_embedding_models(tables) == ("text-b", "vis-b")


# -- scoped video/segment reads ----------------------------------------------


def test_get_videos_by_ids_scopes_to_requested(tmp_path: Path) -> None:
    tables = ensure_tables(connect(tmp_path / "db"))
    for vid in ("aaaa1111aaaa1111", "bbbb2222bbbb2222", "cccc3333cccc3333"):
        upsert_video(tables, _make_video_row(vid))
    got = get_videos_by_ids(tables, ["aaaa1111aaaa1111", "cccc3333cccc3333", "aaaa1111aaaa1111"])
    assert set(got) == {"aaaa1111aaaa1111", "cccc3333cccc3333"}


def test_get_videos_by_ids_empty(tmp_path: Path) -> None:
    tables = ensure_tables(connect(tmp_path / "db"))
    assert get_videos_by_ids(tables, []) == {}


def test_count_segments_for_video(tmp_path: Path) -> None:
    tables = ensure_tables(connect(tmp_path / "db"))
    rng = np.random.default_rng(7)
    vid = "dddd4444dddd4444"
    upsert_segments(tables, [_make_segment_row(vid, i, rng) for i in range(4)])
    assert count_segments_for_video(tables, vid) == 4
    assert count_segments_for_video(tables, "missingmissing00") == 0


def test_read_segment_blob_missing_id_raises(tmp_path: Path) -> None:
    tables = ensure_tables(connect(tmp_path / "db"))
    with pytest.raises(KeyError):
        read_segment_blob(tables, "nope:000000", "clip_bytes")


def test_read_segment_blob_rejects_non_blob_column(tmp_path: Path) -> None:
    tables = ensure_tables(connect(tmp_path / "db"))
    with pytest.raises(ValueError):
        read_segment_blob(tables, "any:000000", "text")
