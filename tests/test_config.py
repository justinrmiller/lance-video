from __future__ import annotations

import pytest
from pydantic import ValidationError

from video_lance.config import FrameSamplingConfig, SegmentationConfig


def test_segmentation_defaults() -> None:
    cfg = SegmentationConfig()
    assert cfg.segment_seconds == 30.0
    assert cfg.overlap_seconds == 0.0
    assert cfg.merge_short_tail is True
    assert cfg.min_tail_seconds == 5.0
    assert cfg.sentence_snap_tolerance_seconds == 0.0


def test_overlap_equals_segment_rejected() -> None:
    with pytest.raises(ValidationError):
        SegmentationConfig(segment_seconds=10.0, overlap_seconds=10.0)


def test_overlap_greater_than_segment_rejected() -> None:
    with pytest.raises(ValidationError):
        SegmentationConfig(segment_seconds=10.0, overlap_seconds=15.0)


def test_negative_segment_rejected() -> None:
    with pytest.raises(ValidationError):
        SegmentationConfig(segment_seconds=-1.0)


def test_zero_segment_rejected() -> None:
    with pytest.raises(ValidationError):
        SegmentationConfig(segment_seconds=0.0)


def test_negative_overlap_rejected() -> None:
    with pytest.raises(ValidationError):
        SegmentationConfig(overlap_seconds=-1.0)


def test_negative_min_tail_rejected() -> None:
    with pytest.raises(ValidationError):
        SegmentationConfig(min_tail_seconds=-1.0)


def test_negative_snap_tolerance_rejected() -> None:
    with pytest.raises(ValidationError):
        SegmentationConfig(sentence_snap_tolerance_seconds=-0.1)


def test_frame_defaults() -> None:
    fc = FrameSamplingConfig()
    assert fc.position == 0.5
    assert fc.jpeg_quality == 85
    assert fc.max_long_edge == 512


def test_frame_position_clamped_low() -> None:
    fc = FrameSamplingConfig(position=-0.5)
    assert fc.position == 0.0


def test_frame_position_clamped_high() -> None:
    fc = FrameSamplingConfig(position=1.5)
    assert fc.position == 1.0


def test_frame_position_in_range_kept() -> None:
    fc = FrameSamplingConfig(position=0.25)
    assert fc.position == 0.25


def test_jpeg_quality_below_min_rejected() -> None:
    with pytest.raises(ValidationError):
        FrameSamplingConfig(jpeg_quality=0)


def test_jpeg_quality_above_max_rejected() -> None:
    with pytest.raises(ValidationError):
        FrameSamplingConfig(jpeg_quality=101)


def test_jpeg_quality_bounds_inclusive() -> None:
    assert FrameSamplingConfig(jpeg_quality=1).jpeg_quality == 1
    assert FrameSamplingConfig(jpeg_quality=100).jpeg_quality == 100


def test_max_long_edge_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        FrameSamplingConfig(max_long_edge=0)
    with pytest.raises(ValidationError):
        FrameSamplingConfig(max_long_edge=-10)
