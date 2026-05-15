from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from video_lance import cli as cli_mod
from video_lance.cli import app

runner = CliRunner()


def test_help_lists_ingest_command() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ingest" in result.stdout


def test_ingest_dry_run_lists_videos(tmp_path: Path, fixture_video: Path) -> None:
    """`--dry-run` walks + reports and writes nothing. It also exercises the
    full CLI argument plumbing without forcing the test to load real
    embedders."""
    root = tmp_path / "root"
    root.mkdir()
    shutil.copy(fixture_video, root / "a.mp4")
    shutil.copy(fixture_video, root / "b.mp4")

    result = runner.invoke(app, ["ingest", str(root), "--dry-run"])
    assert result.exit_code == 0, result.stdout
    assert "discovered 2 video(s)" in result.stdout
    assert "a.mp4" in result.stdout
    assert "b.mp4" in result.stdout


def test_ingest_no_matches_exits_nonzero(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    result = runner.invoke(app, ["ingest", str(empty), "--dry-run"])
    assert result.exit_code == 1
    assert "no videos matched" in result.stdout


def test_ingest_end_to_end_with_injected_fakes(
    tmp_path: Path, fixture_video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Patch the model-loading functions used by the CLI so we can exercise
    the actual ingest command without downloading any weights."""
    from tests._fakes import fake_text_embedder, fake_transcriber, fake_vision_embedder

    monkeypatch.setattr(cli_mod, "get_transcriber", lambda *a, **kw: fake_transcriber())
    monkeypatch.setattr(cli_mod, "get_text_embedder", lambda *a, **kw: fake_text_embedder())
    monkeypatch.setattr(cli_mod, "get_vision_embedder", lambda *a, **kw: fake_vision_embedder())

    root = tmp_path / "src"
    root.mkdir()
    shutil.copy(fixture_video, root / "vid.mp4")
    db_path = tmp_path / "db"

    result = runner.invoke(
        app,
        [
            "ingest",
            str(root),
            "--segment-seconds",
            "2",
            "--no-merge-short-tail",
            "--db-path",
            str(db_path),
            "--frame-max-long-edge",
            "128",
            "--frame-jpeg-quality",
            "60",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "succeeded=1" in result.stdout
    assert "segments_written=5" in result.stdout
    assert db_path.exists()


# ---- search / info / reindex ------------------------------------------------


def _build_ingested_db(
    tmp_path: Path,
    fixture_video: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Run a full CLI ingest with fake models so search/info tests have data."""
    from tests._fakes import fake_text_embedder, fake_transcriber, fake_vision_embedder

    monkeypatch.setattr(cli_mod, "get_transcriber", lambda *a, **kw: fake_transcriber())
    monkeypatch.setattr(cli_mod, "get_text_embedder", lambda *a, **kw: fake_text_embedder())
    monkeypatch.setattr(cli_mod, "get_vision_embedder", lambda *a, **kw: fake_vision_embedder())

    src = tmp_path / "src"
    src.mkdir()
    shutil.copy(fixture_video, src / "vid.mp4")
    db_path = tmp_path / "db"
    result = runner.invoke(
        app,
        [
            "ingest",
            str(src),
            "--segment-seconds",
            "2",
            "--no-merge-short-tail",
            "--db-path",
            str(db_path),
            "--frame-max-long-edge",
            "128",
            "--frame-jpeg-quality",
            "60",
        ],
    )
    assert result.exit_code == 0, result.stdout
    return db_path


def test_info_after_ingest(
    tmp_path: Path, fixture_video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = _build_ingested_db(tmp_path, fixture_video, monkeypatch)
    result = runner.invoke(app, ["info", "--db-path", str(db_path)])
    assert result.exit_code == 0, result.stdout
    assert "videos:" in result.stdout
    assert "segments:" in result.stdout
    assert "embedding models:" in result.stdout


def test_info_missing_db(tmp_path: Path) -> None:
    result = runner.invoke(app, ["info", "--db-path", str(tmp_path / "nope")])
    assert result.exit_code == 2
    assert "no database" in result.stderr


def test_reindex_after_ingest(
    tmp_path: Path, fixture_video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = _build_ingested_db(tmp_path, fixture_video, monkeypatch)
    result = runner.invoke(app, ["reindex", "--db-path", str(db_path)])
    assert result.exit_code == 0, result.stdout
    assert "fts_text:" in result.stdout


def test_search_text_via_cli(
    tmp_path: Path, fixture_video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = _build_ingested_db(tmp_path, fixture_video, monkeypatch)
    # Build FTS first so the hybrid path has both legs.
    runner.invoke(app, ["reindex", "--db-path", str(db_path)])

    result = runner.invoke(
        app,
        ["search", "red.", "--mode", "text", "--limit", "3", "--db-path", str(db_path)],
    )
    assert result.exit_code == 0, result.stdout
    # First result line starts with "1. ["
    assert "1. [" in result.stdout
    assert "open: file://" in result.stdout


def test_search_visual_via_cli(
    tmp_path: Path, fixture_video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = _build_ingested_db(tmp_path, fixture_video, monkeypatch)
    result = runner.invoke(
        app,
        ["search", "red", "--mode", "visual", "--limit", "2", "--db-path", str(db_path)],
    )
    assert result.exit_code == 0, result.stdout
    assert "1. [" in result.stdout


def test_search_multi_via_cli(
    tmp_path: Path, fixture_video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = _build_ingested_db(tmp_path, fixture_video, monkeypatch)
    result = runner.invoke(
        app,
        [
            "search",
            "anything",
            "--mode",
            "multi",
            "--visual-weight",
            "0.4",
            "--limit",
            "3",
            "--db-path",
            str(db_path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "1. [" in result.stdout


def test_search_image_query_via_cli(
    tmp_path: Path, fixture_video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PIL import Image

    db_path = _build_ingested_db(tmp_path, fixture_video, monkeypatch)
    qimg = tmp_path / "q.jpg"
    Image.new("RGB", (32, 32), (255, 0, 0)).save(qimg, format="JPEG")
    result = runner.invoke(
        app,
        [
            "search",
            "",
            "--image",
            str(qimg),
            "--mode",
            "visual",
            "--limit",
            "2",
            "--db-path",
            str(db_path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "1. [" in result.stdout


def test_search_rerank_flag_errors(
    tmp_path: Path, fixture_video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = _build_ingested_db(tmp_path, fixture_video, monkeypatch)
    result = runner.invoke(
        app,
        ["search", "x", "--rerank", "--db-path", str(db_path), "--mode", "text"],
    )
    assert result.exit_code == 2
    assert "not implemented in v1" in result.stderr.lower()


def test_search_unknown_mode_errors(
    tmp_path: Path, fixture_video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = _build_ingested_db(tmp_path, fixture_video, monkeypatch)
    result = runner.invoke(
        app,
        ["search", "x", "--mode", "bogus", "--db-path", str(db_path)],
    )
    assert result.exit_code == 2


def test_search_missing_db(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["search", "x", "--mode", "text", "--db-path", str(tmp_path / "nope")],
    )
    assert result.exit_code == 2
    assert "no database" in result.stderr
