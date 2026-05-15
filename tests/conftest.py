from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pytest

from tests.fixtures.make_fixture import make_fixture
from video_lance.models import TranscriptWord


@pytest.fixture
def make_transcript():
    def _make(words: Iterable[tuple[str, float, float]]) -> list[TranscriptWord]:
        return [TranscriptWord(word=w, start=s, end=e) for w, s, e in words]

    return _make


@pytest.fixture(scope="session")
def fixture_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate the deterministic 10s color-bar fixture once per test session."""
    target_dir = tmp_path_factory.mktemp("video_lance_fixture")
    target = target_dir / "sample.mp4"
    make_fixture(target)
    return target
