#!/usr/bin/env bash
# Quickstart demo: ingest a directory of videos and run one query in each
# search mode. Assumes ffmpeg + uv are on PATH and that `uv sync` has been
# run at least once.
#
# Usage:
#   ./scripts/demo.sh [DIR] [SEGMENT_SECONDS]
#
# DIR defaults to ./videos and SEGMENT_SECONDS defaults to 30. The DB lives
# next to this script's invocation as ./demo.db (a fresh dir each run).

set -euo pipefail

DIR=${1:-./videos}
SEGMENT_SECONDS=${2:-30}
DB=./demo.db

if [ ! -d "$DIR" ]; then
  echo "no directory at $DIR — pass a path with videos as the first arg." >&2
  exit 1
fi

rm -rf "$DB"

echo "==> ingest $DIR (segment_seconds=$SEGMENT_SECONDS) → $DB"
uv run video-lance ingest "$DIR" \
  --segment-seconds "$SEGMENT_SECONDS" \
  --db-path "$DB"

echo
echo "==> info"
uv run video-lance info --db-path "$DB"

echo
echo "==> reindex (idempotent on a freshly-ingested DB)"
uv run video-lance reindex --db-path "$DB"

echo
echo "==> search --mode text 'computer'"
uv run video-lance search 'computer' --mode text --limit 3 --db-path "$DB" || true

echo
echo "==> search --mode visual 'a person at a desk'"
uv run video-lance search 'a person at a desk' --mode visual --limit 3 --db-path "$DB" || true

echo
echo "==> search --mode multi 'artificial intelligence' --visual-weight 0.4"
uv run video-lance search 'artificial intelligence' \
  --mode multi --visual-weight 0.4 --limit 3 --db-path "$DB" || true
