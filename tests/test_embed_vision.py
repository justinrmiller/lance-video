from __future__ import annotations

import hashlib
import io
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from video_lance import embed_vision as embed_vision_mod
from video_lance.embed_vision import (
    VISION_EMBED_DIM,
    SigLIPEmbedder,
    get_vision_embedder,
)


def _solid_image(color: tuple[int, int, int], size: int = 32) -> Image.Image:
    return Image.new("RGB", (size, size), color)


def _hash_to_vec(payload: bytes, dim: int) -> torch.Tensor:
    digest = hashlib.sha256(payload).digest()
    blob = (digest * ((dim // len(digest)) + 1))[:dim]
    arr = np.frombuffer(blob, dtype=np.uint8).astype(np.float32) * 0.13
    return torch.from_numpy(arr.copy())


class _FakeInputs(dict[str, Any]):
    def to(self, _device: str) -> _FakeInputs:
        return self


class _FakeProcessor:
    def __init__(self) -> None:
        self.image_calls: list[int] = []
        self.text_calls: list[list[str]] = []

    def __call__(
        self,
        *,
        images: list[Image.Image] | None = None,
        text: list[str] | None = None,
        return_tensors: str = "pt",
        **_kw: Any,
    ) -> _FakeInputs:
        if images is not None:
            self.image_calls.append(len(images))
            payload = b"|".join(img.tobytes() for img in images)
            # Stash the hashed payload on a deterministic tensor field the model can read.
            return _FakeInputs(
                {
                    "pixel_values": torch.zeros(len(images), 3, 4, 4),
                    "_payload": payload,
                    "_mode": "image",
                }
            )
        if text is not None:
            self.text_calls.append(list(text))
            payload = b"|".join(t.encode() for t in text)
            return _FakeInputs(
                {
                    "input_ids": torch.zeros(len(text), 8, dtype=torch.long),
                    "_payload": payload,
                    "_mode": "text",
                }
            )
        raise ValueError("must pass either images or text")


class _FakeSigLIP:
    def __init__(self) -> None:
        self.image_feature_calls = 0
        self.text_feature_calls = 0

    def to(self, _device: str) -> _FakeSigLIP:
        return self

    def eval(self) -> _FakeSigLIP:
        return self

    def get_image_features(self, **kwargs: Any) -> torch.Tensor:
        self.image_feature_calls += 1
        n = kwargs["pixel_values"].shape[0]
        payload = kwargs.get("_payload", b"")
        # Tag image features with a salt so they don't collide with text features
        # for the same payload (lets us assert encode_image != encode_text routing).
        return torch.stack(
            [_hash_to_vec(b"IMG::" + payload + bytes([i]), VISION_EMBED_DIM) for i in range(n)]
        )

    def get_text_features(self, **kwargs: Any) -> torch.Tensor:
        self.text_feature_calls += 1
        n = kwargs["input_ids"].shape[0]
        payload = kwargs.get("_payload", b"")
        return torch.stack(
            [_hash_to_vec(b"TXT::" + payload + bytes([i]), VISION_EMBED_DIM) for i in range(n)]
        )


@pytest.fixture(autouse=True)
def _reset_cache() -> None:
    embed_vision_mod._reset_cache_for_tests()
    yield
    embed_vision_mod._reset_cache_for_tests()


def _install_fake(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[_FakeSigLIP, _FakeProcessor]]:
    instances: list[tuple[_FakeSigLIP, _FakeProcessor]] = []

    def _factory(model_name: str, device: str) -> tuple[_FakeSigLIP, _FakeProcessor]:
        pair = (_FakeSigLIP(), _FakeProcessor())
        instances.append(pair)
        return pair

    monkeypatch.setattr(embed_vision_mod, "_load_vision_model", _factory)
    return instances


def test_encode_images_shape_and_norm(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake(monkeypatch)
    emb = get_vision_embedder(device="cpu")
    images = [_solid_image((255, 0, 0)), _solid_image((0, 255, 0)), _solid_image((0, 0, 255))]
    out = emb.encode_images(images)
    assert out.shape == (3, VISION_EMBED_DIM)
    assert out.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_encode_single_image_returns_vector(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake(monkeypatch)
    emb = get_vision_embedder(device="cpu")
    out = emb.encode_image(_solid_image((10, 20, 30)))
    assert out.shape == (VISION_EMBED_DIM,)
    assert np.linalg.norm(out) == pytest.approx(1.0, abs=1e-5)


def test_encode_image_accepts_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake(monkeypatch)
    img = _solid_image((40, 50, 60))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    emb = get_vision_embedder(device="cpu")
    out = emb.encode_image(buf.getvalue())
    assert out.shape == (VISION_EMBED_DIM,)


def test_encode_text_shape_and_norm(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake(monkeypatch)
    emb = get_vision_embedder(device="cpu")
    out = emb.encode_text("a person holding a clipboard")
    assert out.shape == (VISION_EMBED_DIM,)
    assert np.linalg.norm(out) == pytest.approx(1.0, abs=1e-5)


def test_encode_text_routes_to_text_features(monkeypatch: pytest.MonkeyPatch) -> None:
    instances = _install_fake(monkeypatch)
    emb = get_vision_embedder(device="cpu")
    emb.encode_text("hello")
    model, proc = instances[0]
    assert model.text_feature_calls == 1
    assert model.image_feature_calls == 0
    assert proc.text_calls == [["hello"]]


def test_encode_image_routes_to_image_features(monkeypatch: pytest.MonkeyPatch) -> None:
    instances = _install_fake(monkeypatch)
    emb = get_vision_embedder(device="cpu")
    emb.encode_image(_solid_image((1, 2, 3)))
    model, proc = instances[0]
    assert model.image_feature_calls == 1
    assert model.text_feature_calls == 0
    assert proc.image_calls == [1]


def test_empty_image_list_skips_model(monkeypatch: pytest.MonkeyPatch) -> None:
    instances = _install_fake(monkeypatch)
    emb = get_vision_embedder(device="cpu")
    out = emb.encode_images([])
    assert out.shape == (0, VISION_EMBED_DIM)
    model, _ = instances[0]
    assert model.image_feature_calls == 0


def test_caches_by_model_and_device(monkeypatch: pytest.MonkeyPatch) -> None:
    instances = _install_fake(monkeypatch)
    a = get_vision_embedder("modelA", device="cpu")
    b = get_vision_embedder("modelA", device="cpu")
    c = get_vision_embedder("modelB", device="cpu")
    assert a is b
    assert a is not c
    assert len(instances) == 2


def test_embedder_remembers_config(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake(monkeypatch)
    emb = get_vision_embedder("siglip-custom", device="cpu")
    assert isinstance(emb, SigLIPEmbedder)
    assert emb.model_name == "siglip-custom"
    assert emb.device == "cpu"


def test_different_images_produce_different_vectors(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake(monkeypatch)
    emb = get_vision_embedder(device="cpu")
    v_red = emb.encode_image(_solid_image((255, 0, 0)))
    v_blue = emb.encode_image(_solid_image((0, 0, 255)))
    assert not np.allclose(v_red, v_blue)
