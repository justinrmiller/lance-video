from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from video_lance import transcribe as transcribe_mod
from video_lance.transcribe import WhisperTranscriber, get_transcriber


@dataclass
class _FakeWord:
    word: str
    start: float | None
    end: float | None


@dataclass
class _FakeSegment:
    words: list[_FakeWord]


@dataclass
class _FakeInfo:
    language: str = "en"


class _FakeModel:
    """Stands in for faster_whisper.WhisperModel; records arguments and returns
    a fixed transcript."""

    def __init__(self, segments: list[_FakeSegment], info: _FakeInfo) -> None:
        self._segments = segments
        self._info = info
        self.calls: list[dict[str, object]] = []

    def transcribe(self, path: str, **kwargs: object) -> tuple[list[_FakeSegment], _FakeInfo]:
        self.calls.append({"path": path, **kwargs})
        return self._segments, self._info


@pytest.fixture(autouse=True)
def _reset_cache() -> None:
    transcribe_mod._reset_cache_for_tests()
    yield
    transcribe_mod._reset_cache_for_tests()


def _install_fake(
    monkeypatch: pytest.MonkeyPatch,
    segments: list[_FakeSegment] | None = None,
    info: _FakeInfo | None = None,
) -> list[_FakeModel]:
    """Patch `_load_model` to return a fresh fake on each call and record
    every instantiation in the returned list."""
    instances: list[_FakeModel] = []

    def _factory(model_name: str, device: str, compute_type: str) -> _FakeModel:
        m = _FakeModel(segments or [], info or _FakeInfo())
        instances.append(m)
        return m

    monkeypatch.setattr(transcribe_mod, "_load_model", _factory)
    return instances


def test_transcribe_returns_words(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    segments = [
        _FakeSegment(
            words=[
                _FakeWord(word="hello", start=0.0, end=0.5),
                _FakeWord(word="world.", start=0.5, end=1.0),
            ]
        )
    ]
    _install_fake(monkeypatch, segments, _FakeInfo(language="en"))

    media = tmp_path / "input.mp4"
    media.write_bytes(b"fake")

    transcriber = get_transcriber()
    result = transcriber.transcribe(media)

    assert [w.word for w in result.words] == ["hello", "world."]
    assert result.words[1].is_sentence_end
    assert result.language == "en"
    assert result.full_text == "hello world."


def test_transcribe_filters_words_with_missing_timestamps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    segments = [
        _FakeSegment(
            words=[
                _FakeWord(word="good", start=0.0, end=0.5),
                _FakeWord(word="dropped", start=None, end=1.0),
                _FakeWord(word="alsogood", start=1.0, end=1.5),
            ]
        )
    ]
    _install_fake(monkeypatch, segments)
    media = tmp_path / "input.mp4"
    media.write_bytes(b"fake")

    result = get_transcriber().transcribe(media)
    assert [w.word for w in result.words] == ["good", "alsogood"]


def test_transcribe_missing_file_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake(monkeypatch)
    with pytest.raises(FileNotFoundError):
        get_transcriber().transcribe(tmp_path / "nope.mp4")


def test_get_transcriber_caches_by_key(monkeypatch: pytest.MonkeyPatch) -> None:
    instances = _install_fake(monkeypatch)

    t1 = get_transcriber("small.en", "cpu", "int8")
    t2 = get_transcriber("small.en", "cpu", "int8")
    assert t1 is t2
    assert len(instances) == 1


def test_get_transcriber_distinct_keys_distinct_instances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances = _install_fake(monkeypatch)

    a = get_transcriber("small.en", "cpu", "int8")
    b = get_transcriber("small.en", "cuda", "int8")
    c = get_transcriber("medium.en", "cpu", "int8")
    d = get_transcriber("small.en", "cpu", "float16")

    assert len({id(a), id(b), id(c), id(d)}) == 4
    assert len(instances) == 4


def test_passes_word_timestamps_kwarg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    instances = _install_fake(monkeypatch)
    media = tmp_path / "input.mp4"
    media.write_bytes(b"fake")
    get_transcriber().transcribe(media, language="en")

    assert len(instances) == 1
    [call] = instances[0].calls
    assert call["word_timestamps"] is True
    assert call["language"] == "en"
    assert call["path"] == str(media)


def test_transcriber_remembers_config(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake(monkeypatch)
    t = get_transcriber("medium.en", "cpu", "int8")
    assert isinstance(t, WhisperTranscriber)
    assert t.model_name == "medium.en"
    assert t.device == "cpu"
    assert t.compute_type == "int8"
