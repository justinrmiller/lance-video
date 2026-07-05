from __future__ import annotations

from video_lance.config import SegmentationConfig
from video_lance.models import TranscriptWord


def compute_segments(
    duration_s: float,
    cfg: SegmentationConfig,
    transcript: list[TranscriptWord] | None = None,
) -> list[tuple[float, float]]:
    """Return list of (start_s, end_s) tuples covering [0, duration_s]."""
    if duration_s <= 0:
        return []
    if duration_s <= cfg.segment_seconds:
        return [(0.0, duration_s)]

    step = cfg.segment_seconds - cfg.overlap_seconds
    assert step > 0, "config validation should have ensured step > 0"

    segments: list[tuple[float, float]] = []
    i = 0
    while True:
        start = i * step
        if start >= duration_s:
            break
        end = min(start + cfg.segment_seconds, duration_s)
        segments.append((start, end))
        if end >= duration_s:
            break
        i += 1

    if cfg.merge_short_tail and len(segments) > 1:
        last_start, last_end = segments[-1]
        if (last_end - last_start) < cfg.min_tail_seconds:
            prev_start, _ = segments[-2]
            segments[-2] = (prev_start, last_end)
            segments.pop()

    if cfg.sentence_snap_tolerance_seconds > 0 and transcript is not None:
        segments = _snap_to_sentences(segments, transcript, cfg.sentence_snap_tolerance_seconds)

    return segments


def _snap_to_sentences(
    segments: list[tuple[float, float]],
    transcript: list[TranscriptWord],
    tol: float,
) -> list[tuple[float, float]]:
    sentence_ends = [w.end for w in transcript if w.is_sentence_end]
    if not sentence_ends:
        return segments

    snapped = [list(seg) for seg in segments]
    n = len(snapped)

    # Snap the end of each non-final segment to the nearest sentence-ending word.
    for i in range(n - 1):
        new_end = _snap_boundary(snapped[i][1], sentence_ends, tol)
        if new_end is not None and new_end > snapped[i][0]:
            snapped[i][1] = new_end

    # Snap the start of each non-first segment to the nearest sentence-ending word.
    for i in range(1, n):
        new_start = _snap_boundary(snapped[i][0], sentence_ends, tol)
        if new_start is not None and new_start < snapped[i][1]:
            snapped[i][0] = new_start

    return [(s, e) for s, e in snapped]


def _snap_boundary(boundary: float, sentence_ends: list[float], tol: float) -> float | None:
    candidates = [t for t in sentence_ends if abs(t - boundary) <= tol]
    if not candidates:
        return None
    return min(candidates, key=lambda t: abs(t - boundary))
