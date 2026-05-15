from __future__ import annotations

import pytest

from video_lance.rerank import RerankNotImplementedError, rerank


def test_rerank_always_raises() -> None:
    with pytest.raises(RerankNotImplementedError):
        rerank([], "anything")


def test_rerank_message_mentions_v1() -> None:
    try:
        rerank([], "x")
    except RerankNotImplementedError as exc:
        assert "v1" in str(exc).lower()
    else:
        raise AssertionError("rerank should have raised")
