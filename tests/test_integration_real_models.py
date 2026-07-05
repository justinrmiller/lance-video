"""Integration tests against the real e5 / SigLIP / Whisper model weights.

**Skipped by default.** Set `VL_INTEGRATION=1` to run:

    VL_INTEGRATION=1 uv run pytest tests/test_integration_real_models.py -v

These tests download multi-GB model weights on first run (~3 GB for e5,
~3.5 GB for SigLIP-SO400M, ~40 MB for Whisper tiny.en) into
`~/.cache/huggingface` and are slow on CPU. They exist for one specific
reason: the unit-test fakes bypass the real model loaders, so missing
transitive native deps (e.g. `sentencepiece` for SigLIP's tokenizer, which
bit us live) don't surface until someone actually runs `video-lance ingest`
on a real video. Running this file catches that class of bug.

Each test is intentionally minimal — load the wrapper, run one tiny
operation, assert basic shape / norm invariants. We don't validate output
quality.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from video_lance import embed_text as embed_text_mod
from video_lance import embed_vision as embed_vision_mod
from video_lance import transcribe as transcribe_mod
from video_lance.embed_text import DEFAULT_TEXT_MODEL, TEXT_EMBED_DIM, get_text_embedder
from video_lance.embed_vision import DEFAULT_VISION_MODEL, VISION_EMBED_DIM, get_vision_embedder
from video_lance.transcribe import get_transcriber

pytestmark = pytest.mark.skipif(
    not os.environ.get("VL_INTEGRATION"),
    reason="integration tests are opt-in; set VL_INTEGRATION=1 to run",
)


@pytest.fixture(autouse=True)
def _reset_caches() -> None:
    # The unit-test fakes leave entries in the per-process model caches under
    # made-up keys like ("fake-e5", "cpu"). Those don't collide with real
    # model names, but we clear anyway so failures here can't be blamed on
    # leftover state from a prior fake-based test run.
    embed_text_mod._reset_cache_for_tests()
    embed_vision_mod._reset_cache_for_tests()
    transcribe_mod._reset_cache_for_tests()
    yield
    embed_text_mod._reset_cache_for_tests()
    embed_vision_mod._reset_cache_for_tests()
    transcribe_mod._reset_cache_for_tests()


# -- text (e5-instruct) ------------------------------------------------------


def test_real_e5_loads_and_encodes_passages_and_query() -> None:
    emb = get_text_embedder(DEFAULT_TEXT_MODEL, device="cpu")

    passages = emb.encode_passages(["hello", "world"])
    assert passages.shape == (2, TEXT_EMBED_DIM)
    assert passages.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(passages, axis=1), 1.0, atol=1e-3)

    query = emb.encode_query("hello world")
    assert query.shape == (TEXT_EMBED_DIM,)
    assert query.dtype == np.float32
    assert np.linalg.norm(query) == pytest.approx(1.0, abs=1e-3)


def test_real_e5_passage_vs_query_for_same_string_differ() -> None:
    """The production guardrail: the asymmetric prompting must actually be
    wired. If `encode_passages("x")` matches `encode_query("x")`, the prefix
    is being silently dropped and retrieval quality collapses without any
    crash to flag it."""
    emb = get_text_embedder(DEFAULT_TEXT_MODEL, device="cpu")
    s = "the quick brown fox"
    p_vec = emb.encode_passages([s])[0]
    q_vec = emb.encode_query(s)
    assert not np.allclose(p_vec, q_vec)


# -- vision (SigLIP-SO400M) --------------------------------------------------


def test_real_siglip_loads_and_encodes_text() -> None:
    """This is the test that would have caught `sentencepiece` missing — the
    SigLIP processor's tokenizer is constructed eagerly on
    `AutoProcessor.from_pretrained`, and it requires sentencepiece to load."""
    emb = get_vision_embedder(DEFAULT_VISION_MODEL, device="cpu")
    vec = emb.encode_text("a photo of a dog")
    assert vec.shape == (VISION_EMBED_DIM,)
    assert vec.dtype == np.float32
    assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-3)


def test_real_siglip_encodes_image() -> None:
    emb = get_vision_embedder(DEFAULT_VISION_MODEL, device="cpu")
    img = Image.new("RGB", (64, 64), (200, 50, 0))
    vec = emb.encode_image(img)
    assert vec.shape == (VISION_EMBED_DIM,)
    assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-3)


def test_real_siglip_encodes_image_batch() -> None:
    emb = get_vision_embedder(DEFAULT_VISION_MODEL, device="cpu")
    images = [
        Image.new("RGB", (64, 64), (200, 50, 0)),
        Image.new("RGB", (64, 64), (0, 100, 200)),
        Image.new("RGB", (64, 64), (50, 200, 50)),
    ]
    vecs = emb.encode_images(images)
    assert vecs.shape == (3, VISION_EMBED_DIM)
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-3)


# -- whisper (faster-whisper) ------------------------------------------------


def test_real_whisper_loads_and_runs_on_fixture(fixture_video: Path) -> None:
    """Loads `tiny.en` (~40 MB) and transcribes the 10s color-bar fixture.

    The fixture is a 440 Hz sine tone — Whisper may produce no words or
    nonsense. We only verify the wrapper runs end-to-end without crashing
    and returns a `Transcript` shape.
    """
    transcriber = get_transcriber("tiny.en", device="cpu", compute_type="int8")
    result = transcriber.transcribe(fixture_video)
    # `Transcript` model: `words` is always a list, even if empty.
    assert hasattr(result, "words")
    assert isinstance(result.words, list)
