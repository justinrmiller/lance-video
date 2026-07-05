from __future__ import annotations

import pytest

from video_lance.config import SegmentationConfig
from video_lance.segmenter import compute_segments


def test_duration_zero_returns_empty() -> None:
    assert compute_segments(0.0, SegmentationConfig()) == []


def test_duration_negative_returns_empty() -> None:
    assert compute_segments(-5.0, SegmentationConfig()) == []


def test_duration_less_than_segment() -> None:
    cfg = SegmentationConfig(segment_seconds=30.0)
    assert compute_segments(10.0, cfg) == [(0.0, 10.0)]


def test_duration_equals_segment() -> None:
    cfg = SegmentationConfig(segment_seconds=30.0)
    assert compute_segments(30.0, cfg) == [(0.0, 30.0)]


def test_exact_multiple_no_overlap() -> None:
    cfg = SegmentationConfig(segment_seconds=30.0, overlap_seconds=0.0, merge_short_tail=False)
    assert compute_segments(60.0, cfg) == [(0.0, 30.0), (30.0, 60.0)]


def test_with_overlap() -> None:
    cfg = SegmentationConfig(
        segment_seconds=30.0,
        overlap_seconds=10.0,
        merge_short_tail=False,
    )
    segs = compute_segments(60.0, cfg)
    starts = [s for s, _ in segs]
    assert starts[:3] == [0.0, 20.0, 40.0]
    assert segs[0] == (0.0, 30.0)
    assert segs[1] == (20.0, 50.0)
    # Last segment clipped to duration.
    assert segs[-1][1] == 60.0


def test_short_tail_merged() -> None:
    cfg = SegmentationConfig(
        segment_seconds=10.0,
        overlap_seconds=0.0,
        merge_short_tail=True,
        min_tail_seconds=5.0,
    )
    # duration 22 → raw windows (0,10),(10,20),(20,22). Tail of 2s < 5s → merge.
    segs = compute_segments(22.0, cfg)
    assert segs == [(0.0, 10.0), (10.0, 22.0)]


def test_short_tail_kept_when_disabled() -> None:
    cfg = SegmentationConfig(
        segment_seconds=10.0,
        overlap_seconds=0.0,
        merge_short_tail=False,
        min_tail_seconds=5.0,
    )
    segs = compute_segments(22.0, cfg)
    assert segs == [(0.0, 10.0), (10.0, 20.0), (20.0, 22.0)]


def test_short_tail_above_threshold_kept() -> None:
    cfg = SegmentationConfig(
        segment_seconds=10.0,
        overlap_seconds=0.0,
        merge_short_tail=True,
        min_tail_seconds=5.0,
    )
    # tail of 7s >= 5s → keep
    segs = compute_segments(27.0, cfg)
    assert segs == [(0.0, 10.0), (10.0, 20.0), (20.0, 27.0)]


def test_single_segment_no_merge_attempt() -> None:
    cfg = SegmentationConfig(
        segment_seconds=30.0,
        merge_short_tail=True,
        min_tail_seconds=5.0,
    )
    # duration 3 < segment 30 → trivial case; no merge to attempt.
    assert compute_segments(3.0, cfg) == [(0.0, 3.0)]


@pytest.mark.parametrize(
    ("duration", "segment", "overlap", "merge"),
    [
        (60.0, 30.0, 0.0, True),
        (60.0, 30.0, 10.0, True),
        (61.5, 7.5, 1.5, True),
        (100.0, 17.3, 4.2, False),
        (10.0, 10.0, 0.0, True),
        (0.5, 30.0, 0.0, True),
    ],
)
def test_no_zero_or_negative_segments_ever(
    duration: float, segment: float, overlap: float, merge: bool
) -> None:
    cfg = SegmentationConfig(
        segment_seconds=segment, overlap_seconds=overlap, merge_short_tail=merge
    )
    segs = compute_segments(duration, cfg)
    for start, end in segs:
        assert end > start, f"degenerate segment ({start}, {end})"


def test_sentence_snap_basic(make_transcript) -> None:
    cfg = SegmentationConfig(
        segment_seconds=30.0,
        overlap_seconds=0.0,
        merge_short_tail=False,
        sentence_snap_tolerance_seconds=1.0,
    )
    # Sentence-ending word ends at 30.4, within 1.0s of boundary 30.0.
    transcript = make_transcript([("hello.", 30.0, 30.4)])
    segs = compute_segments(60.0, cfg, transcript)
    # Boundary at 30.0 snapped to 30.4 on both sides (no overlap).
    assert segs[0][1] == pytest.approx(30.4)
    assert segs[1][0] == pytest.approx(30.4)


def test_sentence_snap_picks_closest(make_transcript) -> None:
    cfg = SegmentationConfig(
        segment_seconds=30.0,
        overlap_seconds=0.0,
        merge_short_tail=False,
        sentence_snap_tolerance_seconds=2.0,
    )
    transcript = make_transcript(
        [
            ("first.", 28.0, 28.5),  # distance 1.5
            ("second.", 30.7, 30.9),  # distance 0.9 (closest)
            ("third.", 31.7, 31.8),  # distance 1.8
        ]
    )
    segs = compute_segments(60.0, cfg, transcript)
    assert segs[0][1] == pytest.approx(30.9)


def test_sentence_snap_no_candidate_falls_back(make_transcript) -> None:
    cfg = SegmentationConfig(
        segment_seconds=30.0,
        overlap_seconds=0.0,
        merge_short_tail=False,
        sentence_snap_tolerance_seconds=1.0,
    )
    # Only sentence-ending word is well outside the tolerance window.
    transcript = make_transcript([("far.", 45.0, 45.2)])
    segs = compute_segments(60.0, cfg, transcript)
    assert segs[0][1] == pytest.approx(30.0)
    assert segs[1][0] == pytest.approx(30.0)


def test_sentence_snap_does_not_create_degenerate(make_transcript) -> None:
    cfg = SegmentationConfig(
        segment_seconds=10.0,
        overlap_seconds=0.0,
        merge_short_tail=False,
        sentence_snap_tolerance_seconds=15.0,  # absurdly large to invite inversion
    )
    # A sentence-ending word at 0.1s would, if snapped onto the 10.0 boundary, push it before
    # the start of segment 1 (0.0). This must be rejected.
    transcript = make_transcript([("oops.", 0.0, 0.1)])
    segs = compute_segments(20.0, cfg, transcript)
    for start, end in segs:
        assert end > start


def test_sentence_snap_tolerance_zero_disables(make_transcript) -> None:
    cfg = SegmentationConfig(
        segment_seconds=30.0,
        overlap_seconds=0.0,
        merge_short_tail=False,
        sentence_snap_tolerance_seconds=0.0,
    )
    transcript = make_transcript([("hello.", 30.0, 30.4)])
    segs = compute_segments(60.0, cfg, transcript)
    assert segs[0][1] == 30.0
    assert segs[1][0] == 30.0


def test_sentence_snap_with_no_transcript() -> None:
    cfg = SegmentationConfig(
        segment_seconds=30.0,
        overlap_seconds=0.0,
        merge_short_tail=False,
        sentence_snap_tolerance_seconds=1.0,
    )
    segs = compute_segments(60.0, cfg, transcript=None)
    assert segs == [(0.0, 30.0), (30.0, 60.0)]


def test_sentence_snap_ignores_non_sentence_words(make_transcript) -> None:
    cfg = SegmentationConfig(
        segment_seconds=30.0,
        overlap_seconds=0.0,
        merge_short_tail=False,
        sentence_snap_tolerance_seconds=1.0,
    )
    # All words inside the tolerance window, but none ends a sentence.
    transcript = make_transcript([("just", 29.8, 30.0), ("words", 30.2, 30.5)])
    segs = compute_segments(60.0, cfg, transcript)
    assert segs[0][1] == 30.0
    assert segs[1][0] == 30.0


def test_first_start_and_last_end_not_snapped(make_transcript) -> None:
    cfg = SegmentationConfig(
        segment_seconds=30.0,
        overlap_seconds=0.0,
        merge_short_tail=False,
        sentence_snap_tolerance_seconds=2.0,
    )
    transcript = make_transcript(
        [
            ("beginning.", 0.0, 0.4),  # near start of first segment
            ("ending.", 59.5, 59.8),  # near end of last segment
        ]
    )
    segs = compute_segments(60.0, cfg, transcript)
    assert segs[0][0] == 0.0
    assert segs[-1][1] == 60.0
