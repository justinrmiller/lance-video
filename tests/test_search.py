from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from tests._fakes import fake_text_embedder, fake_transcriber, fake_vision_embedder
from video_lance import search, store
from video_lance.config import Config, FrameSamplingConfig, SegmentationConfig
from video_lance.pipeline import process_video
from video_lance.search import (
    _rrf_fuse,
    db_info,
    ensure_indexes,
    format_timestamp,
    search_multi,
    search_text,
    search_visual,
)

# -- pure helpers -------------------------------------------------------------


def test_format_timestamp_zero() -> None:
    assert format_timestamp(0.0) == "00:00:00"


def test_format_timestamp_minute_boundary() -> None:
    assert format_timestamp(60.0) == "00:01:00"
    assert format_timestamp(59.99) == "00:00:59"


def test_format_timestamp_hours() -> None:
    assert format_timestamp(754.5) == "00:12:34"
    assert format_timestamp(3601.0) == "01:00:01"


def test_format_timestamp_negative_clamped() -> None:
    assert format_timestamp(-5.0) == "00:00:00"


def test_rrf_fuse_equal_weights() -> None:
    a = [{"segment_id": "x"}, {"segment_id": "y"}]
    b = [{"segment_id": "y"}, {"segment_id": "z"}]
    fused = _rrf_fuse([a, b], k=60, limit=3, labels=("a", "b"))
    fused_dict = {sid: (score, comp) for sid, score, comp in fused}

    # x is rank 0 in list a only.
    # y is rank 1 in a and rank 0 in b — total contributions add up.
    # z is rank 1 in b only.
    expected_x = 1.0 / 61
    expected_y = 1.0 / 62 + 1.0 / 61
    expected_z = 1.0 / 62
    assert fused_dict["x"][0] == pytest.approx(expected_x)
    assert fused_dict["y"][0] == pytest.approx(expected_y)
    assert fused_dict["z"][0] == pytest.approx(expected_z)
    # y has contributions from both labels.
    assert "a" in fused_dict["y"][1]
    assert "b" in fused_dict["y"][1]


def test_rrf_fuse_weighted() -> None:
    a = [{"segment_id": "x"}]
    b = [{"segment_id": "x"}]
    # Both lists rank x at position 0; equal weight should give total 2/(60+1).
    [(sid, score, comp)] = _rrf_fuse([a, b], weights=[0.5, 0.5], k=60, limit=1)
    assert sid == "x"
    assert score == pytest.approx((0.5 + 0.5) / 61)


def test_rrf_fuse_weight_length_mismatch_rejected() -> None:
    with pytest.raises(ValueError):
        _rrf_fuse([[{"segment_id": "x"}]], weights=[0.5, 0.5], limit=1)


def test_rrf_fuse_labels_length_mismatch_rejected() -> None:
    with pytest.raises(ValueError):
        _rrf_fuse(
            [[{"segment_id": "x"}], [{"segment_id": "y"}]],
            labels=("only_one",),
            limit=1,
        )


# -- integration fixture: a small ingested DB --------------------------------


@pytest.fixture
def ingested_db(tmp_path: Path, fixture_video: Path) -> store.StoreTables:
    """Ingest the 10s fixture into a fresh DB with five 2-second segments.

    Each segment's text is one of: 'red.', 'green.', 'blue.', 'yellow.',
    'magenta.' (set by `_FakeWhisperModel`). Embeddings are deterministic
    hashes of the input strings via the shared fakes.
    """
    cfg = Config(
        segmentation=SegmentationConfig(segment_seconds=2.0, merge_short_tail=False),
        frames=FrameSamplingConfig(max_long_edge=128, jpeg_quality=60),
        db_path=tmp_path / "db",
    )
    db = store.connect(cfg.db_path)
    tables = store.ensure_tables(db)
    store.set_embedding_models(
        tables,
        text_embed_model="fake-e5",
        vision_embed_model="fake-siglip",
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
    return tables


# -- search modes -------------------------------------------------------------


def test_search_text_returns_hits_with_filled_fields(ingested_db: store.StoreTables) -> None:
    # FTS needs the inverted index; build it.
    ensure_indexes(ingested_db)
    hits = search_text(ingested_db, fake_text_embedder(), "red.", limit=5)
    assert hits
    assert all(h.segment_id for h in hits)
    assert all(h.source_path.endswith(".mp4") for h in hits)
    assert all(h.start_s < h.end_s for h in hits)
    # Hybrid path should record per-source contributions.
    assert any(h.components for h in hits)


def test_search_text_fts_finds_exact_word(ingested_db: store.StoreTables) -> None:
    ensure_indexes(ingested_db)
    hits = search_text(ingested_db, fake_text_embedder(), "red", limit=5)
    # Top hit's text should contain the queried token (FTS contributes
    # alongside the deterministic-but-arbitrary vector ranking).
    top_texts = [h.text.lower() for h in hits]
    assert any("red" in t for t in top_texts)


def test_search_text_falls_back_to_vector_when_fts_missing(
    ingested_db: store.StoreTables,
) -> None:
    # Don't call ensure_indexes — FTS is unavailable. search_text must catch
    # the underlying error and return vector-only results.
    hits = search_text(ingested_db, fake_text_embedder(), "anything", limit=3)
    assert hits  # vector brute-force still works
    for h in hits:
        # Single-source fallback records the vector score under "vector".
        assert "vector" in h.components


def test_search_visual_text_query(ingested_db: store.StoreTables) -> None:
    hits = search_visual(ingested_db, fake_vision_embedder(), query="anything", limit=5)
    assert len(hits) == 5
    assert all(h.components.get("visual") is not None for h in hits)


def test_search_visual_with_image_path(ingested_db: store.StoreTables, tmp_path: Path) -> None:
    img_path = tmp_path / "query.jpg"
    Image.new("RGB", (32, 32), (255, 0, 0)).save(img_path, format="JPEG")
    hits = search_visual(ingested_db, fake_vision_embedder(), image=img_path, limit=3)
    assert len(hits) == 3


def test_search_visual_with_pil_image(ingested_db: store.StoreTables) -> None:
    img = Image.new("RGB", (32, 32), (255, 0, 0))
    hits = search_visual(ingested_db, fake_vision_embedder(), image=img, limit=2)
    assert len(hits) == 2


def test_search_visual_with_jpeg_bytes(ingested_db: store.StoreTables) -> None:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (0, 0, 255)).save(buf, format="JPEG")
    hits = search_visual(ingested_db, fake_vision_embedder(), image=buf.getvalue(), limit=2)
    assert len(hits) == 2


def test_search_visual_requires_query_or_image(ingested_db: store.StoreTables) -> None:
    with pytest.raises(ValueError):
        search_visual(ingested_db, fake_vision_embedder(), limit=5)
    with pytest.raises(ValueError):
        search_visual(
            ingested_db, fake_vision_embedder(), query="x", image=Image.new("RGB", (1, 1)), limit=5
        )


def test_search_multi_blends_text_and_visual(ingested_db: store.StoreTables) -> None:
    hits = search_multi(
        ingested_db,
        fake_text_embedder(),
        fake_vision_embedder(),
        "anything",
        limit=5,
        visual_weight=0.4,
    )
    assert len(hits) == 5
    for h in hits:
        assert "text" in h.components
        assert "visual" in h.components


def test_search_multi_visual_weight_extremes(ingested_db: store.StoreTables) -> None:
    text_only = search_multi(
        ingested_db,
        fake_text_embedder(),
        fake_vision_embedder(),
        "anything",
        limit=5,
        visual_weight=0.0,
    )
    visual_only = search_multi(
        ingested_db,
        fake_text_embedder(),
        fake_vision_embedder(),
        "anything",
        limit=5,
        visual_weight=1.0,
    )
    # When visual_weight=0, every fused score comes from the text leg.
    for h in text_only:
        assert h.components.get("visual", 0.0) == 0.0
    for h in visual_only:
        assert h.components.get("text", 0.0) == 0.0


def test_search_multi_rejects_out_of_range_weight(ingested_db: store.StoreTables) -> None:
    with pytest.raises(ValueError):
        search_multi(
            ingested_db,
            fake_text_embedder(),
            fake_vision_embedder(),
            "x",
            visual_weight=1.5,
        )


# -- SQL filter ---------------------------------------------------------------


def test_search_filter_restricts_results(ingested_db: store.StoreTables) -> None:
    hits = search_visual(
        ingested_db,
        fake_vision_embedder(),
        query="x",
        limit=10,
        sql_filter="idx >= 3",
    )
    assert hits
    assert all(h.idx >= 3 for h in hits)


# -- deep link ----------------------------------------------------------------


def test_hit_deep_link_format(ingested_db: store.StoreTables) -> None:
    hits = search_visual(ingested_db, fake_vision_embedder(), query="x", limit=1)
    link = hits[0].deep_link()
    assert link.startswith("file://")
    assert "#t=" in link
    assert hits[0].time_range().count(":") == 4  # HH:MM:SS–HH:MM:SS


# -- ensure_indexes & info ----------------------------------------------------


def test_ensure_indexes_on_small_table(ingested_db: store.StoreTables) -> None:
    """5 rows is far below IVF_PQ's 256-row training floor; the vector index
    build should be reported as skipped rather than crashing."""
    status = ensure_indexes(ingested_db)
    assert status.fts_text == "ok"
    assert status.vec_text.startswith(("ok", "skipped", "exists"))
    assert status.vec_visual.startswith(("ok", "skipped", "exists"))


def test_ensure_indexes_idempotent(ingested_db: store.StoreTables) -> None:
    first = ensure_indexes(ingested_db)
    second = ensure_indexes(ingested_db)
    # Second pass should not error; FTS already exists.
    assert second.fts_text in {"ok", "exists"}
    assert first.fts_text == "ok"


def test_ensure_indexes_replace_rebuilds(ingested_db: store.StoreTables) -> None:
    ensure_indexes(ingested_db)
    status = ensure_indexes(ingested_db, replace=True)
    # `replace=True` should not refuse on the second call.
    assert status.fts_text in {"ok", "exists"}


def test_db_info_reports_models_and_indexes(ingested_db: store.StoreTables) -> None:
    ensure_indexes(ingested_db)
    info = db_info(ingested_db, db_path=Path("/some/path"))
    assert info.db_path == "/some/path"
    assert info.videos == 1
    assert info.segments == 5
    assert info.text_embed_model == "fake-e5"
    assert info.vision_embed_model == "fake-siglip"
    assert any("text" in idx for idx in info.segment_indexes)


def test_db_info_empty_db(tmp_path: Path) -> None:
    db = store.connect(tmp_path / "empty.db")
    tables = store.ensure_tables(db)
    info = search.db_info(tables, db_path=tmp_path / "empty.db")
    assert info.videos == 0
    assert info.segments == 0
    assert info.text_embed_model is None
    assert info.vision_embed_model is None
