from __future__ import annotations

import fnmatch
from collections.abc import Iterable
from pathlib import Path

from video_lance.config import DEFAULT_INCLUDE


def _any_match(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def walk(
    root: Path,
    *,
    include: Iterable[str] = DEFAULT_INCLUDE,
    exclude: Iterable[str] = (),
) -> list[Path]:
    """Walk `root` recursively and return matching video files.

    A file is returned iff its name matches at least one `include` glob and
    matches none of the `exclude` globs. Matching is done with `fnmatch` on
    the file's basename, so callers pass patterns like `"*.mp4"`. Results are
    sorted for determinism.

    Non-existent `root` returns an empty list (the CLI prefers to log this
    rather than blow up).
    """
    root = Path(root)
    if not root.exists():
        return []
    if root.is_file():
        return [root] if _matches(root, include, exclude) else []

    include_patterns = tuple(include)
    exclude_patterns = tuple(exclude)
    results: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if _matches(path, include_patterns, exclude_patterns):
            results.append(path)
    return sorted(results)


def _matches(
    path: Path,
    include: Iterable[str],
    exclude: Iterable[str],
) -> bool:
    name = path.name
    if not _any_match(name, include):
        return False
    return not _any_match(name, exclude)
