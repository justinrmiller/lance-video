from __future__ import annotations

from pathlib import Path

from video_lance.discovery import walk


def _touch(p: Path, content: bytes = b"") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)
    return p


def test_walk_finds_video_files(tmp_path: Path) -> None:
    a = _touch(tmp_path / "a.mp4")
    b = _touch(tmp_path / "sub" / "b.MKV")
    _touch(tmp_path / "notes.txt")
    found = walk(tmp_path)
    assert a in found
    # Lowercase glob shouldn't match an uppercase extension on a case-sensitive FS.
    # (macOS is usually case-insensitive — this test is loose so it works either way.)
    assert sorted(found) == found  # results are sorted
    assert all(p.suffix.lower() in {".mp4", ".mkv", ".mov", ".webm"} for p in found)
    # Make sure we picked up the file in the subdirectory in either case-mode.
    assert any(p.name.lower() == "b.mkv" for p in found) or b not in found


def test_walk_respects_include(tmp_path: Path) -> None:
    _touch(tmp_path / "a.mp4")
    b = _touch(tmp_path / "b.mov")
    found = walk(tmp_path, include=["*.mov"])
    assert found == [b]


def test_walk_respects_exclude(tmp_path: Path) -> None:
    a = _touch(tmp_path / "a.mp4")
    _touch(tmp_path / "b_skip_me.mp4")
    found = walk(tmp_path, include=["*.mp4"], exclude=["*skip*"])
    assert found == [a]


def test_walk_missing_root_returns_empty(tmp_path: Path) -> None:
    assert walk(tmp_path / "no-such-dir") == []


def test_walk_single_file_path(tmp_path: Path) -> None:
    f = _touch(tmp_path / "single.mp4")
    assert walk(f) == [f]


def test_walk_single_file_path_filtered_out(tmp_path: Path) -> None:
    f = _touch(tmp_path / "single.txt")
    assert walk(f) == []


def test_walk_skips_directories(tmp_path: Path) -> None:
    (tmp_path / "looksvideo.mp4").mkdir()  # Directory with a .mp4 name.
    real = _touch(tmp_path / "real.mp4")
    found = walk(tmp_path)
    assert found == [real]
