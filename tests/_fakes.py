"""Shared fakes for tests that exercise pipeline code end-to-end.

The real wrapper classes (`WhisperTranscriber`, `E5Embedder`, `SigLIPEmbedder`)
all take their underlying model as a constructor argument. Tests instantiate
the real wrapper class with a fake model — so the wrapper code (caching,
prefix logic, normalization, dispatch) is exercised, only the model itself is
stubbed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image

from video_lance.embed_text import E5Embedder
from video_lance.embed_vision import SigLIPEmbedder
from video_lance.schema import TEXT_EMBED_DIM, VISION_EMBED_DIM
from video_lance.transcribe import WhisperTranscriber


def _pseudo_text_vec(text: str) -> np.ndarray:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    blob = (digest * ((TEXT_EMBED_DIM // len(digest)) + 1))[:TEXT_EMBED_DIM]
    return np.frombuffer(blob, dtype=np.uint8).astype(np.float32) * 0.37


class _FakeSentenceTransformer:
    def encode(self, texts: list[str], *, convert_to_numpy: bool = True) -> np.ndarray:
        return np.stack([_pseudo_text_vec(t) for t in texts])


def fake_text_embedder() -> E5Embedder:
    return E5Embedder("fake-e5", "cpu", _FakeSentenceTransformer())


def _hash_to_tensor(payload: bytes, dim: int) -> torch.Tensor:
    digest = hashlib.sha256(payload).digest()
    blob = (digest * ((dim // len(digest)) + 1))[:dim]
    arr = np.frombuffer(blob, dtype=np.uint8).astype(np.float32) * 0.13
    return torch.from_numpy(arr.copy())


class _FakeProcessorInputs(dict[str, Any]):
    def to(self, _device: str) -> _FakeProcessorInputs:
        return self


class _FakeSigLIPProcessor:
    def __call__(
        self,
        *,
        images: list[Image.Image] | None = None,
        text: list[str] | None = None,
        return_tensors: str = "pt",
        **_kw: Any,
    ) -> _FakeProcessorInputs:
        if images is not None:
            payload = b"|".join(img.tobytes() for img in images)
            return _FakeProcessorInputs(
                {
                    "pixel_values": torch.zeros(len(images), 3, 4, 4),
                    "_payload": payload,
                }
            )
        if text is not None:
            payload = b"|".join(t.encode() for t in text)
            return _FakeProcessorInputs(
                {
                    "input_ids": torch.zeros(len(text), 8, dtype=torch.long),
                    "_payload": payload,
                }
            )
        raise ValueError("must pass images or text")


class _FakeSigLIPModel:
    def to(self, _device: str) -> _FakeSigLIPModel:
        return self

    def eval(self) -> _FakeSigLIPModel:
        return self

    def get_image_features(self, **kwargs: Any) -> torch.Tensor:
        n = kwargs["pixel_values"].shape[0]
        payload = kwargs.get("_payload", b"")
        return torch.stack(
            [_hash_to_tensor(b"IMG::" + payload + bytes([i]), VISION_EMBED_DIM) for i in range(n)]
        )

    def get_text_features(self, **kwargs: Any) -> torch.Tensor:
        n = kwargs["input_ids"].shape[0]
        payload = kwargs.get("_payload", b"")
        return torch.stack(
            [_hash_to_tensor(b"TXT::" + payload + bytes([i]), VISION_EMBED_DIM) for i in range(n)]
        )


def fake_vision_embedder() -> SigLIPEmbedder:
    return SigLIPEmbedder("fake-siglip", "cpu", _FakeSigLIPModel(), _FakeSigLIPProcessor())


# -- whisper ------------------------------------------------------------------


@dataclass
class _FakeWhisperWord:
    word: str
    start: float | None
    end: float | None


@dataclass
class _FakeWhisperSegment:
    words: list[_FakeWhisperWord]


@dataclass
class _FakeWhisperInfo:
    language: str = "en"


class _FakeWhisperModel:
    """Returns a fixed transcript regardless of input audio. The transcript
    covers the full fixture duration with five short sentence-ending words so
    `compute_segments` with sentence-snap has something to look at."""

    def transcribe(
        self,
        path: str,
        **_kwargs: Any,
    ) -> tuple[list[_FakeWhisperSegment], _FakeWhisperInfo]:
        sentences = ["red.", "green.", "blue.", "yellow.", "magenta."]
        words: list[_FakeWhisperWord] = []
        for i, w in enumerate(sentences):
            t = i * 2.0
            words.append(_FakeWhisperWord(word=w, start=t + 0.1, end=t + 1.0))
        return [_FakeWhisperSegment(words=words)], _FakeWhisperInfo()


def fake_transcriber() -> WhisperTranscriber:
    return WhisperTranscriber("fake-whisper", "cpu", "default", _FakeWhisperModel())
