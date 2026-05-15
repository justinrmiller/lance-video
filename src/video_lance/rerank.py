from __future__ import annotations

from typing import Any


class RerankNotImplementedError(NotImplementedError):
    """Raised by `rerank(...)` while cross-encoder rerank is out of scope.

    Kept as a distinct subclass so the CLI can catch it and print a nicer
    message than a bare NotImplementedError.
    """


def rerank(hits: list[Any], query: str) -> list[Any]:
    """Cross-encoder rerank — not implemented in v1.

    PLAN §13 lists this as an "out-of-scope hook": the function and the
    `--rerank` flag exist so the wiring is right, but the actual reranker
    isn't built yet.
    """
    raise RerankNotImplementedError(
        "--rerank is not implemented in v1; see PLAN §13 (out-of-scope hooks)."
    )
