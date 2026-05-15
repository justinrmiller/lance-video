from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Protocol

from video_lance.models import Transcript, TranscriptWord


class _WhisperWord(Protocol):
    word: str
    start: float
    end: float


class _WhisperSegment(Protocol):
    words: list[_WhisperWord] | None


def _load_model(model_name: str, device: str, compute_type: str) -> Any:
    # Imported lazily so importing video_lance.transcribe doesn't drag faster-whisper into every
    # process (the CLI and UI both import the package; model load cost is unwanted at import time).
    from faster_whisper import WhisperModel

    return WhisperModel(model_name, device=device, compute_type=compute_type)


class WhisperTranscriber:
    """Thin wrapper around faster_whisper.WhisperModel.

    The underlying model is loaded once per (model_name, device, compute_type)
    triple via `get_transcriber`; instances of this class are cheap value
    objects that hold a reference to a shared model.
    """

    def __init__(self, model_name: str, device: str, compute_type: str, model: Any) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self._model = model

    def transcribe(self, path: Path, *, language: str | None = None) -> Transcript:
        if not path.exists():
            raise FileNotFoundError(path)

        segments_iter, info = self._model.transcribe(
            str(path),
            word_timestamps=True,
            language=language,
        )

        words: list[TranscriptWord] = []
        for seg in segments_iter:
            seg_words = getattr(seg, "words", None) or []
            for w in seg_words:
                if w.start is None or w.end is None:
                    continue
                words.append(
                    TranscriptWord(word=str(w.word), start=float(w.start), end=float(w.end))
                )

        detected_lang = getattr(info, "language", None) if info is not None else None
        return Transcript(words=words, language=detected_lang)


def map_text_to_window(transcript: Transcript, start_s: float, end_s: float) -> str:
    """Return the joined transcript text for words overlapping [start_s, end_s).

    A word "overlaps" the window if its [w.start, w.end) range intersects
    [start_s, end_s). Whitespace is normalized; empty results return ''.
    """
    parts: list[str] = []
    for w in transcript.words:
        if w.start < end_s and w.end > start_s:
            stripped = w.word.strip()
            if stripped:
                parts.append(stripped)
    return " ".join(parts)


_cache_lock = threading.Lock()
_cache: dict[tuple[str, str, str], WhisperTranscriber] = {}


def get_transcriber(
    model_name: str = "small.en",
    device: str = "auto",
    compute_type: str = "default",
) -> WhisperTranscriber:
    """Return a process-cached transcriber for the given configuration.

    Calls with the same (model_name, device, compute_type) tuple return the
    same instance; the underlying model is loaded exactly once.
    """
    key = (model_name, device, compute_type)
    with _cache_lock:
        existing = _cache.get(key)
        if existing is not None:
            return existing
        model = _load_model(model_name, device, compute_type)
        transcriber = WhisperTranscriber(model_name, device, compute_type, model)
        _cache[key] = transcriber
        return transcriber


def _reset_cache_for_tests() -> None:
    with _cache_lock:
        _cache.clear()
