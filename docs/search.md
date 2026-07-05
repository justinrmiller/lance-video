# Search

All retrieval lives in `search.py` and returns a list of `SearchHit` dataclasses
(`segment_id`, `video_id`, paths, `start_s`/`end_s`, `text`, `score`, and a
per-source `components` dict). `SearchHit.deep_link()` renders a
`file://…#t=start,end` URL; `time_range()` renders `HH:MM:SS–HH:MM:SS`.

## The three modes

### `text` — hybrid e5 + full-text
`search_text` runs two independent searches and fuses them with RRF:

1. **Vector**: e5 `encode_query` → cosine search over `text_embedding`.
2. **FTS**: BM25 over the `text` column (needs the inverted index).

It's implemented as two searches + RRF rather than LanceDB's built-in
`query_type="hybrid"` so behavior is consistent regardless of which indexes
exist. If the FTS index is missing, it logs a warning and **falls back to
vector-only** results (scored by cosine similarity).

### `visual` — SigLIP cross-modal
`search_visual` takes exactly one of a text `query` or an `image` (PIL / bytes /
path). The query is encoded with the SigLIP text or image tower and searched
against `visual_embedding`. Scores are cosine similarity.

### `multi` — weighted blend
`search_multi` encodes the query with both e5 (→ `text_embedding`) and SigLIP
text (→ `visual_embedding`), then RRF-fuses the two rankings with weights
`[1 - visual_weight, visual_weight]` (default `visual_weight = 0.4`).

## Reciprocal-rank fusion (RRF)

`_rrf_fuse` keys on `segment_id`. For a list `i`, an item at rank `r` (0 = best)
contributes `weights[i] / (k + r + 1)`, with `k = 60` (the convention from the
original RRF paper). Contributions across lists are summed; items missing from a
list contribute `0` from it. Each hit keeps its per-source contributions in
`components` (e.g. `{"vector": …, "fts": …}` or `{"text": …, "visual": …}`).

## Score normalization

Raw RRF scores are tiny and rank-derived (≈0.01–0.03), which looks far worse than
the cosine similarities (`≈ -1..1`) that the single-source paths surface — even
for the top hit. So the hybrid (`text`) and `multi` paths pass their fused
results through `_normalize_fused_scores`, which divides by the max so the best
hit sits at `1.0` and the relative gaps are preserved. The surfaced `score` is
therefore comparable in range across modes; the raw contributions remain
untouched in `components`.

## SQL filters

Every search accepts an optional `sql_filter` — a raw LanceDB `WHERE` expression
(e.g. `duration_s > 60`, `idx >= 3`) applied to the segment search before
fusion. It's passed straight through, so it's a power-user feature scoped to
your own store.

## Video resolution

Hits carry `source_path` / `relative_path`, which come from the `videos` table.
Rather than scan the whole table per query, `_fetch_video_index` fetches only the
`video_id`s referenced by the current result rows via
`store.get_videos_by_ids`.

## Indexing

`ensure_indexes(tables, replace=False)` builds three indexes on `segments` and
returns an `IndexStatus` (`ok` / `exists` / `skipped: …`):

- **FTS** (`text_idx`) on the `text` column — required for the `text` mode's BM25
  leg; works at any row count.
- **IVF vector** indexes on `text_embedding` (`text_embedding_idx`) and
  `visual_embedding` (`visual_embedding_idx`) — accelerate vector search on large
  tables. `IVF_FLAT` under 100k rows, `IVF_PQ` above; `num_partitions =
  max(2, floor(sqrt(n)))`.

Status is decided from **observable state**, not exception message text: existing
indexes are found via `list_indices()` (reported `exists` unless `replace=True`),
and vector builds are gated on `count_rows()` against a `_MIN_VECTOR_INDEX_ROWS`
floor (256) — below which they report `skipped` because IVF training needs the
rows. Only a residual, genuinely-unexpected error falls through to
`skipped: {exc}`. This makes the status robust to LanceDB wording changes across
versions.

Vector indexes are optional: brute-force search works on unindexed columns, so a
small store still searches correctly (just without ANN acceleration).

`reindex` (CLI) / **Rebuild indexes** (UI) call `ensure_indexes(replace=True)` to
drop and rebuild all three.

## `db_info`

`db_info` powers `video-lance info` and the UI's Database tab: DB path, video and
segment counts, the persisted embedding-model identifiers, and a label per
segment-table index (`"name on columns"`).
