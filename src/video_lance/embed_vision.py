from __future__ import annotations

import io
import threading
from typing import Any

import numpy as np
from PIL import Image

from video_lance.device import resolve_device
from video_lance.schema import VISION_EMBED_DIM

__all__ = [
    "DEFAULT_VISION_MODEL",
    "SigLIPEmbedder",
    "VISION_EMBED_DIM",
    "get_vision_embedder",
]

DEFAULT_VISION_MODEL = "google/siglip2-so400m-patch14-384"


def _load_vision_model(model_name: str, device: str) -> tuple[Any, Any]:
    # Lazy import: transformers + torch are large; defer until first use.
    from transformers import AutoModel, AutoProcessor

    model = AutoModel.from_pretrained(model_name).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_name)  # type: ignore[no-untyped-call]
    return model, processor


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return np.asarray(arr / norms, dtype=np.float32)


def _features_to_tensor(features: Any) -> Any:
    """Normalize the return of `get_image_features` / `get_text_features`.

    Older transformers versions return a raw tensor; current versions
    (≥5.x for SigLIP) return a `BaseModelOutputWithPooling` whose
    `pooler_output` carries the embedding we want. Handle both so the
    wrapper isn't tied to a specific transformers minor version.
    """
    if hasattr(features, "pooler_output") and features.pooler_output is not None:
        return features.pooler_output
    if hasattr(features, "last_hidden_state"):
        # Defensive: a configuration that returns the full sequence without
        # a pooler. Mean-pool ourselves rather than throw.
        return features.last_hidden_state.mean(dim=1)
    return features


def _to_pil(image: Image.Image | bytes) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(io.BytesIO(image)).convert("RGB")


class SigLIPEmbedder:
    """Wrapper around `google/siglip2-so400m-patch14-384` (or any compatible
    SigLIP / SigLIP 2 checkpoint).

    Image and text encoders share an embedding space, so a text query encoded
    with `encode_text` can be compared directly against image vectors from
    `encode_image(s)`. Outputs are L2-normalized.

    SigLIP 2 (Tschannen et al., 2025) keeps the same SO400M architecture and
    1152-dim output as SigLIP 1, so the schema, vector indexes, and this
    wrapper are unchanged. The improvements are in training (better
    multilingual support, NaFlex-style variable-aspect-ratio variants,
    improved localization). The class name stays `SigLIPEmbedder` rather
    than spelling out the version — it works with both.
    """

    def __init__(self, model_name: str, device: str, model: Any, processor: Any) -> None:
        self.model_name = model_name
        self.device = device
        self._model = model
        self._processor = processor

    def encode_images(self, images: list[Image.Image | bytes]) -> np.ndarray:
        if not images:
            return np.empty((0, VISION_EMBED_DIM), dtype=np.float32)

        import torch

        pil_images = [_to_pil(img) for img in images]
        inputs = self._processor(images=pil_images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            raw = self._model.get_image_features(**inputs)
        features = _features_to_tensor(raw)
        arr = features.detach().cpu().numpy()
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return _l2_normalize(arr)

    def encode_image(self, image: Image.Image | bytes) -> np.ndarray:
        return np.asarray(self.encode_images([image])[0], dtype=np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        import torch

        inputs = self._processor(
            text=[text], return_tensors="pt", padding="max_length", truncation=True
        ).to(self.device)
        with torch.no_grad():
            raw = self._model.get_text_features(**inputs)
        features = _features_to_tensor(raw)
        arr = features.detach().cpu().numpy()
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        normalized: np.ndarray = _l2_normalize(arr)
        return np.asarray(normalized[0], dtype=np.float32)


_cache_lock = threading.Lock()
_cache: dict[tuple[str, str], SigLIPEmbedder] = {}


def get_vision_embedder(
    model_name: str = DEFAULT_VISION_MODEL,
    device: str = "auto",
) -> SigLIPEmbedder:
    """Return a process-cached SigLIP embedder. Loads lazily on first call."""
    resolved = resolve_device(device)
    key = (model_name, resolved)
    with _cache_lock:
        cached = _cache.get(key)
        if cached is not None:
            return cached
        model, processor = _load_vision_model(model_name, resolved)
        emb = SigLIPEmbedder(model_name, resolved, model, processor)
        _cache[key] = emb
        return emb


def _reset_cache_for_tests() -> None:
    with _cache_lock:
        _cache.clear()
