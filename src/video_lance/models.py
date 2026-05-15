from __future__ import annotations

from pydantic import BaseModel


class TranscriptWord(BaseModel):
    word: str
    start: float
    end: float

    @property
    def is_sentence_end(self) -> bool:
        stripped = self.word.strip()
        return stripped.endswith((".", "?", "!"))


class Transcript(BaseModel):
    words: list[TranscriptWord]
    language: str | None = None

    @property
    def full_text(self) -> str:
        return " ".join(w.word.strip() for w in self.words if w.word.strip())


class VideoMeta(BaseModel):
    duration_s: float
    fps: float
    width: int
    height: int
    codec: str
    size_bytes: int
