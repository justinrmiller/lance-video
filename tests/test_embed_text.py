from __future__ import annotations

import hashlib

import numpy as np
import pytest

from video_lance import embed_text as embed_text_mod
from video_lance.embed_text import (
    DEFAULT_E5_TASK,
    TEXT_EMBED_DIM,
    E5Embedder,
    _format_query,
    get_text_embedder,
)


def _pseudo_vec(text: str, dim: int) -> np.ndarray:
    """Deterministic non-unit-norm vector derived from `text`."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    needed = dim
    blob = (digest * ((needed // len(digest)) + 1))[:needed]
    vec = np.frombuffer(blob, dtype=np.uint8).astype(np.float32)
    # Scale away from unit-norm so the wrapper's normalization is observable.
    return vec * 0.37


class _FakeSentenceTransformer:
    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = device
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str], *, convert_to_numpy: bool = True) -> np.ndarray:
        self.calls.append(list(texts))
        return np.stack([_pseudo_vec(t, TEXT_EMBED_DIM) for t in texts])


@pytest.fixture(autouse=True)
def _reset_cache() -> None:
    embed_text_mod._reset_cache_for_tests()
    yield
    embed_text_mod._reset_cache_for_tests()


def _install_fake(monkeypatch: pytest.MonkeyPatch) -> list[_FakeSentenceTransformer]:
    instances: list[_FakeSentenceTransformer] = []

    def _factory(model_name: str, device: str) -> _FakeSentenceTransformer:
        m = _FakeSentenceTransformer(model_name, device)
        instances.append(m)
        return m

    monkeypatch.setattr(embed_text_mod, "_load_text_model", _factory)
    return instances


def test_encode_passages_shape_and_norm(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake(monkeypatch)
    emb = get_text_embedder(device="cpu")
    out = emb.encode_passages(["alpha", "beta", "gamma"])
    assert out.shape == (3, TEXT_EMBED_DIM)
    assert out.dtype == np.float32
    norms = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-5)


def test_encode_query_shape_and_norm(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake(monkeypatch)
    emb = get_text_embedder(device="cpu")
    out = emb.encode_query("how does ranking work")
    assert out.shape == (TEXT_EMBED_DIM,)
    assert out.dtype == np.float32
    assert np.linalg.norm(out) == pytest.approx(1.0, abs=1e-5)


def test_passage_and_query_differ_for_same_string(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sanity check: the prefix logic is wired so passage("X") != query("X").
    This is the guardrail against silent prefix bugs that destroy retrieval
    quality.
    """
    _install_fake(monkeypatch)
    emb = get_text_embedder(device="cpu")
    s = "the quick brown fox"
    passage_vec = emb.encode_passages([s])[0]
    query_vec = emb.encode_query(s)
    assert not np.allclose(passage_vec, query_vec)


def test_query_task_changes_output(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake(monkeypatch)
    emb = get_text_embedder(device="cpu")
    s = "anything"
    default_task = emb.encode_query(s)
    custom_task = emb.encode_query(s, task="Find related code examples")
    assert not np.allclose(default_task, custom_task)


def test_format_query_uses_default_task() -> None:
    assert _format_query("hi", None) == f"Instruct: {DEFAULT_E5_TASK}\nQuery: hi"


def test_format_query_uses_custom_task() -> None:
    assert _format_query("hi", "Do something") == "Instruct: Do something\nQuery: hi"


def test_empty_passages_skips_model(monkeypatch: pytest.MonkeyPatch) -> None:
    instances = _install_fake(monkeypatch)
    emb = get_text_embedder(device="cpu")
    out = emb.encode_passages([])
    assert out.shape == (0, TEXT_EMBED_DIM)
    assert instances[0].calls == []


def test_passages_sent_without_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    instances = _install_fake(monkeypatch)
    emb = get_text_embedder(device="cpu")
    emb.encode_passages(["raw text"])
    assert instances[0].calls == [["raw text"]]


def test_query_sent_with_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    instances = _install_fake(monkeypatch)
    emb = get_text_embedder(device="cpu")
    emb.encode_query("raw text")
    [(call,)] = instances[0].calls
    assert call.startswith("Instruct: ")
    assert call.endswith("\nQuery: raw text")


def test_caches_by_model_and_device(monkeypatch: pytest.MonkeyPatch) -> None:
    instances = _install_fake(monkeypatch)
    a = get_text_embedder("modelA", device="cpu")
    b = get_text_embedder("modelA", device="cpu")
    c = get_text_embedder("modelB", device="cpu")
    assert a is b
    assert a is not c
    assert len(instances) == 2


def test_embedder_remembers_config(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake(monkeypatch)
    emb = get_text_embedder("custom-e5", device="cpu")
    assert isinstance(emb, E5Embedder)
    assert emb.model_name == "custom-e5"
    assert emb.device == "cpu"


def test_zero_vector_does_not_explode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Defensive: if the underlying model ever returned a zero vector, we
    shouldn't NaN out. Patch the fake to return zeros."""

    class _ZeroST:
        def encode(self, texts: list[str], *, convert_to_numpy: bool = True) -> np.ndarray:
            return np.zeros((len(texts), TEXT_EMBED_DIM), dtype=np.float32)

    monkeypatch.setattr(embed_text_mod, "_load_text_model", lambda *_a, **_kw: _ZeroST())
    emb = get_text_embedder(device="cpu")
    out = emb.encode_passages(["x"])
    assert out.shape == (1, TEXT_EMBED_DIM)
    assert not np.isnan(out).any()
