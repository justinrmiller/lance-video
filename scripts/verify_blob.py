"""Verify that the segments table is using LanceDB blob columns (Blob V1).

Run against an existing video-lance DB:

    uv run python scripts/verify_blob.py --db-path ./video-lance.db

"Blob V1" here means the legacy `lance-encoding:blob` field-metadata encoding
that stores large binary payloads out-of-line (Lance file format <= 2.1). The
newer adaptive Blob V2 (file format 2.2+) uses a different opt-in — the
`lance.blob.v2` Arrow extension type — and is not yet reachable through
`lancedb` 0.33, whose create_table pins format 2.1. When that lands, this
script is the natural place to assert the upgraded encoding.

Checks performed:

1. The `clip_bytes` and `keyframe_jpeg` fields on the segments table carry
   the `lance-encoding:blob = true` metadata (this is the Blob V1 opt-in).
2. The underlying Lance dataset exposes `take_blobs`, the blob read API.
3. We can fetch the bytes of one row's `clip_bytes` and `keyframe_jpeg` via
   `take_blobs` and they're non-empty.

Exits 0 with "Blob V1 OK" on success, non-zero with the failing assertion
otherwise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from video_lance import store

EXPECTED_BLOB_COLUMNS = ("clip_bytes", "keyframe_jpeg")
BLOB_METADATA_KEY = b"lance-encoding:blob"
BLOB_METADATA_VALUE = b"true"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=Path("./video-lance.db"))
    args = parser.parse_args()

    if not args.db_path.exists():
        print(f"error: no database at {args.db_path}", file=sys.stderr)
        return 2

    db = store.connect(args.db_path)
    tables = store.ensure_tables(db)
    ds = tables.segments.to_lance()

    # 1. Schema metadata opt-in.
    schema = ds.schema
    for column in EXPECTED_BLOB_COLUMNS:
        field = schema.field(column)
        meta = field.metadata or {}
        if meta.get(BLOB_METADATA_KEY) != BLOB_METADATA_VALUE:
            print(
                f"FAIL: field {column!r} is missing "
                f"`lance-encoding:blob=true` metadata (got {dict(meta)!r})",
                file=sys.stderr,
            )
            return 1
        print(f"  ✓ {column}: lance-encoding:blob=true")

    # 2/3. Blob read path against a real row.
    if not hasattr(ds, "take_blobs"):
        print("FAIL: lance dataset has no take_blobs — blob read API missing", file=sys.stderr)
        return 1

    row_arrow = ds.to_table(columns=[], limit=1, with_row_id=True)
    row_ids = row_arrow.column("_rowid").to_pylist()
    if not row_ids:
        print("warn: segments table is empty — schema check passed but no rows to read")
        print("Blob V1 OK")
        return 0

    rid = row_ids[0]
    for column in EXPECTED_BLOB_COLUMNS:
        blobs = ds.take_blobs(column, ids=[rid])
        if not blobs:
            print(f"FAIL: take_blobs returned no blob for {column} row {rid}", file=sys.stderr)
            return 1
        data = bytes(blobs[0].read())
        if not data:
            print(f"FAIL: blob for {column} row {rid} is empty", file=sys.stderr)
            return 1
        print(f"  ✓ {column}: take_blobs returned {len(data)} bytes")

    print("Blob V1 OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
