from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from video_lance import store
from video_lance.embed_text import E5Embedder
from video_lance.embed_vision import SigLIPEmbedder

logger = logging.getLogger(__name__)

# Names we give the indexes when we create them. `list_indices()` reflects
# these so `info` can report cleanly.
FTS_INDEX_NAME = "text_idx"
TEXT_VEC_INDEX_NAME = "text_embedding_idx"
VISUAL_VEC_INDEX_NAME = "visual_embedding_idx"

# This module plumbs the text / visual / multi search modes over LanceDB.


# -- result types -------------------------------------------------------------


@dataclass
class SearchHit:
    segment_id: str
    video_id: str
    source_path: str
    relative_path: str
    idx: int
    start_s: float
    end_s: float
    text: str
    score: float
    components: dict[str, float] = field(default_factory=dict)

    def deep_link(self) -> str:
        """A file://...#t=start,end URL the CLI prints."""
        return f"file://{self.source_path}#t={self.start_s:.1f},{self.end_s:.1f}"

    def time_range(self) -> str:
        return f"{format_timestamp(self.start_s)}–{format_timestamp(self.end_s)}"


def format_timestamp(seconds: float) -> str:
    """`754.5` → `'00:12:34'`. Used in the CLI's per-hit output line."""
    s = max(0.0, float(seconds))
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


# -- low-level helpers --------------------------------------------------------


def _vector_search(
    table: Any,
    column: str,
    qvec: np.ndarray,
    limit: int,
    sql_filter: str | None,
) -> list[dict[str, Any]]:
    builder = table.search(qvec.astype(np.float32), vector_column_name=column)
    if sql_filter:
        builder = builder.where(sql_filter)
    rows: list[dict[str, Any]] = builder.limit(limit).to_arrow().to_pylist()
    return rows


def _fts_search(
    table: Any,
    query: str,
    limit: int,
    sql_filter: str | None,
) -> list[dict[str, Any]]:
    builder = table.search(query, query_type="fts")
    if sql_filter:
        builder = builder.where(sql_filter)
    rows: list[dict[str, Any]] = builder.limit(limit).to_arrow().to_pylist()
    return rows


def _rrf_fuse(
    ranked_lists: list[list[dict[str, Any]]],
    *,
    weights: list[float] | None = None,
    k: int = 60,
    limit: int,
    labels: tuple[str, ...] | None = None,
) -> list[tuple[str, float, dict[str, float]]]:
    """Reciprocal-rank-fuse `ranked_lists` keyed on `segment_id`.

    Each list is treated as a ranking (rank 0 = best). The contribution from
    list `i` for an item at rank `r` is `weights[i] / (k + r + 1)`. Items not
    in a list contribute 0 from that list. Returns the top-`limit` items as
    `(segment_id, fused_score, per_label_components)`.

    `k=60` is the convention from the original RRF paper; `weights` defaults
    to equal. `labels` provides names for the per-source score keys (e.g.
    `("vector", "fts")`); without them keys are `src_0`, `src_1`, ...
    """
    n = len(ranked_lists)
    if weights is None:
        weights = [1.0] * n
    if len(weights) != n:
        raise ValueError(f"weights length {len(weights)} != lists length {n}")
    if labels is not None and len(labels) != n:
        raise ValueError(f"labels length {len(labels)} != lists length {n}")
    label_names = labels if labels is not None else tuple(f"src_{i}" for i in range(n))

    scores: dict[str, float] = {}
    components: dict[str, dict[str, float]] = {}
    for label, rows, w in zip(label_names, ranked_lists, weights, strict=True):
        for rank, row in enumerate(rows):
            sid = row["segment_id"]
            contribution = w / (k + rank + 1)
            scores[sid] = scores.get(sid, 0.0) + contribution
            components.setdefault(sid, {})[label] = contribution

    items = sorted(scores.items(), key=lambda kv: -kv[1])[:limit]
    return [(sid, score, components[sid]) for sid, score in items]


def _fetch_video_index(tables: store.StoreTables) -> dict[str, dict[str, Any]]:
    rows: list[dict[str, Any]] = tables.videos.search().limit(1_000_000).to_arrow().to_pylist()
    return {r["video_id"]: r for r in rows}


def _hit_from_row(
    row: dict[str, Any],
    *,
    videos: dict[str, dict[str, Any]],
    score: float,
    components: dict[str, float] | None = None,
) -> SearchHit:
    video = videos.get(row["video_id"], {})
    return SearchHit(
        segment_id=row["segment_id"],
        video_id=row["video_id"],
        source_path=str(video.get("source_path", "")),
        relative_path=str(video.get("relative_path", "")),
        idx=int(row["idx"]),
        start_s=float(row["start_s"]),
        end_s=float(row["end_s"]),
        text=str(row.get("text") or ""),
        score=score,
        components=components or {},
    )


def _cosine_similarity_from_distance(distance: float) -> float:
    """LanceDB returns `_distance` in [0, 2] for cosine on unit vectors.
    Convert back to cosine similarity in [-1, 1]."""
    return 1.0 - float(distance)


# -- search modes -------------------------------------------------------------


def search_text(
    tables: store.StoreTables,
    text_embedder: E5Embedder,
    query: str,
    *,
    limit: int = 10,
    sql_filter: str | None = None,
) -> list[SearchHit]:
    """e5 vector search + FTS BM25 over `text`, fused via RRF.

    This is "hybrid" retrieval. We implement it as two separate searches
    fused with RRF rather than going through LanceDB's `query_type='hybrid'`
    so we get consistent behavior regardless of which indexes exist.
    """
    qvec = text_embedder.encode_query(query)
    vec_rows = _vector_search(tables.segments, "text_embedding", qvec, limit * 3, sql_filter)

    fts_rows: list[dict[str, Any]] = []
    try:
        fts_rows = _fts_search(tables.segments, query, limit * 3, sql_filter)
    except Exception as exc:  # noqa: BLE001 - want to fall back rather than fail the whole search
        logger.warning("FTS unavailable, falling back to vector-only text search: %s", exc)

    videos = _fetch_video_index(tables)
    rows_by_sid: dict[str, dict[str, Any]] = {r["segment_id"]: r for r in vec_rows + fts_rows}

    if fts_rows:
        fused = _rrf_fuse(
            [vec_rows, fts_rows],
            weights=[0.5, 0.5],
            limit=limit,
            labels=("vector", "fts"),
        )
        return [
            _hit_from_row(rows_by_sid[sid], videos=videos, score=score, components=components)
            for sid, score, components in fused
        ]

    return [
        _hit_from_row(
            r,
            videos=videos,
            score=_cosine_similarity_from_distance(r.get("_distance", 1.0)),
            components={"vector": _cosine_similarity_from_distance(r.get("_distance", 1.0))},
        )
        for r in vec_rows[:limit]
    ]


def search_visual(
    tables: store.StoreTables,
    vision_embedder: SigLIPEmbedder,
    *,
    query: str | None = None,
    image: Image.Image | bytes | Path | None = None,
    limit: int = 10,
    sql_filter: str | None = None,
) -> list[SearchHit]:
    """Cross-modal SigLIP search against `visual_embedding`.

    Exactly one of `query` (a text string, encoded with the SigLIP text tower)
    or `image` (a PIL image / bytes / path, encoded with the SigLIP image
    tower) must be supplied.
    """
    if (query is None) == (image is None):
        raise ValueError("provide exactly one of `query` or `image`")

    if image is not None:
        if isinstance(image, Path):
            image = image.read_bytes()
        qvec = vision_embedder.encode_image(image)
    else:
        assert query is not None
        qvec = vision_embedder.encode_text(query)

    rows = _vector_search(tables.segments, "visual_embedding", qvec, limit, sql_filter)
    videos = _fetch_video_index(tables)
    return [
        _hit_from_row(
            r,
            videos=videos,
            score=_cosine_similarity_from_distance(r.get("_distance", 1.0)),
            components={"visual": _cosine_similarity_from_distance(r.get("_distance", 1.0))},
        )
        for r in rows
    ]


def search_multi(
    tables: store.StoreTables,
    text_embedder: E5Embedder,
    vision_embedder: SigLIPEmbedder,
    query: str,
    *,
    limit: int = 10,
    visual_weight: float = 0.4,
    sql_filter: str | None = None,
) -> list[SearchHit]:
    """Weighted RRF blend of text-mode and visual-mode rankings.

    `visual_weight` is the weight given to the SigLIP-text→frame ranking;
    `1 - visual_weight` is the weight on the e5-text→text ranking. The
    default is 0.4.
    """
    if not 0.0 <= visual_weight <= 1.0:
        raise ValueError(f"visual_weight {visual_weight} must be in [0, 1]")

    text_qvec = text_embedder.encode_query(query)
    vis_qvec = vision_embedder.encode_text(query)

    text_rows = _vector_search(tables.segments, "text_embedding", text_qvec, limit * 3, sql_filter)
    vis_rows = _vector_search(tables.segments, "visual_embedding", vis_qvec, limit * 3, sql_filter)

    fused = _rrf_fuse(
        [text_rows, vis_rows],
        weights=[1.0 - visual_weight, visual_weight],
        limit=limit,
        labels=("text", "visual"),
    )
    videos = _fetch_video_index(tables)
    rows_by_sid: dict[str, dict[str, Any]] = {r["segment_id"]: r for r in text_rows + vis_rows}
    return [
        _hit_from_row(rows_by_sid[sid], videos=videos, score=score, components=components)
        for sid, score, components in fused
    ]


# -- index management ---------------------------------------------------------


@dataclass
class IndexStatus:
    fts_text: str
    vec_text: str
    vec_visual: str


def ensure_indexes(tables: store.StoreTables, *, replace: bool = False) -> IndexStatus:
    """Build (or rebuild) the segment indexes the search paths benefit from.

    - FTS on `text` is required for the hybrid text path's BM25 leg.
    - Vector indexes on `text_embedding` / `visual_embedding` speed up search
      on large tables but aren't required — brute-force search works on
      unindexed columns.

    On a small table (< 256 rows) LanceDB refuses to train an IVF_PQ index;
    we catch that and report `skipped` so search still works. With `replace`,
    existing indexes are dropped and rebuilt; otherwise we leave them alone.
    """
    segments = tables.segments

    # FTS — works on any row count.
    try:
        segments.create_fts_index("text", replace=replace, name=FTS_INDEX_NAME)
        fts_status = "ok"
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).lower()
        if "already exists" in msg:
            fts_status = "exists"
        else:
            fts_status = f"skipped: {exc}"
            logger.warning("FTS index build failed: %s", exc)

    n = segments.count_rows()
    # IVF_FLAT for small N, IVF_PQ for large; num_partitions = max(8, sqrt(n)).
    num_partitions = max(2, int(math.sqrt(max(n, 1))))
    index_type = "IVF_FLAT" if n < 100_000 else "IVF_PQ"

    def _build_vec_index(column: str, index_name: str) -> str:
        try:
            segments.create_index(
                metric="cosine",
                vector_column_name=column,
                num_partitions=num_partitions,
                index_type=index_type,
                replace=replace,
                name=index_name,
            )
            return "ok"
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            if "already exists" in msg and not replace:
                return "exists"
            if "not enough" in msg or "too few" in msg or "requires" in msg:
                return f"skipped (need more rows; have {n})"
            logger.warning("vector index on %s failed: %s", column, exc)
            return f"skipped: {exc}"

    vec_text_status = _build_vec_index("text_embedding", TEXT_VEC_INDEX_NAME)
    vec_visual_status = _build_vec_index("visual_embedding", VISUAL_VEC_INDEX_NAME)

    return IndexStatus(
        fts_text=fts_status,
        vec_text=vec_text_status,
        vec_visual=vec_visual_status,
    )


# -- info ---------------------------------------------------------------------


@dataclass
class DBInfo:
    db_path: str
    videos: int
    segments: int
    text_embed_model: str | None
    vision_embed_model: str | None
    segment_indexes: list[str]


def db_info(tables: store.StoreTables, *, db_path: Path | None = None) -> DBInfo:
    text_model, vision_model = store.get_embedding_models(tables)
    seg_indices_raw = tables.segments.list_indices()
    seg_indices = [_index_label(i) for i in seg_indices_raw]
    return DBInfo(
        db_path=str(db_path) if db_path else "",
        videos=int(tables.videos.count_rows()),
        segments=int(tables.segments.count_rows()),
        text_embed_model=text_model,
        vision_embed_model=vision_model,
        segment_indexes=seg_indices,
    )


def _index_label(idx: Any) -> str:
    name = getattr(idx, "name", "") or ""
    columns = getattr(idx, "columns", None) or []
    cols = ",".join(str(c) for c in columns)
    if name and cols:
        return f"{name} on {cols}"
    return name or cols or repr(idx)
