from __future__ import annotations

import threading
from typing import Any

import numpy as np

from video_lance.device import resolve_device
from video_lance.schema import TEXT_EMBED_DIM

__all__ = [
    "DEFAULT_E5_TASK",
    "DEFAULT_TEXT_MODEL",
    "E5Embedder",
    "TEXT_EMBED_DIM",
    "get_text_embedder",
]

DEFAULT_TEXT_MODEL = "intfloat/multilingual-e5-large-instruct"
DEFAULT_E5_TASK = "Given a web search query, retrieve relevant passages that answer the query"


def _load_text_model(model_name: str, device: str) -> Any:
    # Lazy import: sentence-transformers pulls torch + transformers, and we don't want every
    # import of video_lance to drag those in.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device)


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    # Avoid division by zero — leave zero vectors as-is.
    norms = np.where(norms == 0.0, 1.0, norms)
    return np.asarray(arr / norms, dtype=np.float32)


def _format_query(query: str, task: str | None) -> str:
    actual_task = task if task is not None else DEFAULT_E5_TASK
    return f"Instruct: {actual_task}\nQuery: {query}"


class E5Embedder:
    """Wrapper around `intfloat/multilingual-e5-large-instruct`.

    Asymmetric prompting: passages are encoded as-is, queries are wrapped in
    `Instruct: {task}\\nQuery: {query}`. Output vectors are L2-normalized so
    cosine similarity equals dot product.
    """

    def __init__(self, model_name: str, device: str, model: Any) -> None:
        self.model_name = model_name
        self.device = device
        self._model = model

    def encode_passages(self, passages: list[str]) -> np.ndarray:
        if not passages:
            return np.empty((0, TEXT_EMBED_DIM), dtype=np.float32)
        raw = self._model.encode(list(passages), convert_to_numpy=True)
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return _l2_normalize(arr)

    def encode_query(self, query: str, task: str | None = None) -> np.ndarray:
        prefixed = _format_query(query, task)
        raw = self._model.encode([prefixed], convert_to_numpy=True)
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        normalized: np.ndarray = _l2_normalize(arr)
        return np.asarray(normalized[0], dtype=np.float32)


_cache_lock = threading.Lock()
_cache: dict[tuple[str, str], E5Embedder] = {}


def get_text_embedder(
    model_name: str = DEFAULT_TEXT_MODEL,
    device: str = "auto",
) -> E5Embedder:
    """Return a process-cached e5 embedder. Models load lazily on first call."""
    resolved = resolve_device(device)
    key = (model_name, resolved)
    with _cache_lock:
        cached = _cache.get(key)
        if cached is not None:
            return cached
        model = _load_text_model(model_name, resolved)
        emb = E5Embedder(model_name, resolved, model)
        _cache[key] = emb
        return emb


def _reset_cache_for_tests() -> None:
    with _cache_lock:
        _cache.clear()
