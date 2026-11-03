from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import REPO_ROOT

from face_wall import __version__, read_image
from face_wall.cli import main


@pytest.fixture
def in_project(face_dir: Path, monkeypatch) -> Path:
    """Run from a folder that has ./faces, like the README tells users to."""
    monkeypatch.chdir(face_dir.parent)
    return face_dir.parent


# Every spelling the old python-fire command line accepted for "5 columns, 6 rows, out.jpg".
@pytest.mark.parametrize(
    "argv",
    [
        ["5", "6", "out.jpg"],
        ["-x", "5", "-y", "6", "-save_path", "out.jpg"],
        ["--x=5", "--y=6", "--save-path", "out.jpg"],
        ["--save_path=out.jpg", "5", "6"],
        ["out.jpg", "-x", "5", "-y", "6"],
        ["5", "-y", "6", "out.jpg"],
    ],
)
def test_old_argument_spellings_still_work(in_project: Path, argv: list[str]) -> None:
    assert main(argv) == 0
    assert read_image(in_project / "out.jpg").shape == (6 * 72, 5 * 72, 3)


def test_defaults_match_the_old_script(in_project: Path) -> None:
    assert main(["3", "2"]) == 0
    assert read_image(in_project / "wall.jpg").shape == (2 * 72, 3 * 72, 3)


def test_new_options(in_project: Path, tmp_path: Path) -> None:
    faces = in_project / "faces"
    other = tmp_path / "elsewhere"
    faces.rename(other)
    argv = ["4", "2", "a.png", "--faces", str(other), "--size", "20x30", "--seed", "3"]
    assert main(argv) == 0
    assert main([*argv[:2], "b.png", *argv[3:]]) == 0
    first, second = read_image(in_project / "a.png"), read_image(in_project / "b.png")
    assert first.shape == (2 * 30, 4 * 20, 3)
    assert np.array_equal(first, second)


def test_reports_progress_on_stderr(in_project: Path, capsys) -> None:
    assert main(["5", "6"]) == 0
    err = capsys.readouterr().err
    assert "face-wall: loaded 3 faces from ./faces" in err
    assert "face-wall: saved a wall of 5 x 6 faces (360 x 432 px) to wall.jpg" in err


@pytest.mark.parametrize(
    "argv, message",
    [
        (["5"], "missing Y"),
        (["five", "6"], "X: expected a whole number"),
        (["-x", "five", "6"], "expected a whole number"),
        (["5", "6", "out.jpg", "extra"], "unrecognized arguments: extra"),
        (["-x", "5", "6", "out.jpg", "extra"], "unexpected extra argument: extra"),
        (["--size", "big", "5", "6"], "expected N or WIDTHxHEIGHT"),
    ],
)
def test_usage_errors_exit_with_2(in_project: Path, capsys, argv, message) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(argv)
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv, message",
    [
        (["5", "6", "--faces", "nowhere"], "face-wall: error: face folder not found: nowhere"),
        (["0", "6"], "face-wall: error: columns (x) must be a whole number of at least 1"),
        (["5", "6", "no/such/dir/wall.jpg"], "face-wall: error: output folder does not exist"),
    ],
)
def test_problems_are_reported_without_a_traceback(in_project: Path, capsys, argv, message) -> None:
    assert main(argv) == 1
    err = capsys.readouterr().err
    assert message in err
    assert "Traceback" not in err


def test_help_lists_options_and_examples(capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    for text in ["X Y [SAVE_PATH]", "--crop-faces", "--seed", "--size", "examples:"]:
        assert text in out


def test_main_py_still_runs_as_a_script(in_project: Path) -> None:
    # Regression: main.py imported loguru, which was never listed as a dependency, so a fresh
    # install crashed with ModuleNotFoundError before doing anything.
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "main.py"), "2", "2", "script.jpg"],
        capture_output=True,
        text=True,
        cwd=in_project,
    )
    assert result.returncode == 0, result.stderr
    assert read_image(in_project / "script.jpg").shape == (144, 144, 3)


def test_python_dash_m_reports_the_version() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "face_wall", "--version"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert result.stdout.strip() == f"face-wall {__version__}"


def test_main_py_keeps_the_old_python_names(face_dir: Path, tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("old_main", REPO_ROOT / "main.py")
    old_main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old_main)
    assert old_main.FACE_FOLDER == "./faces"
    assert len(old_main.get_faces(str(face_dir))) == 3
    old_main.generate_face_wall(2, 1, str(tmp_path / "old.jpg"), faces=face_dir)
    assert read_image(tmp_path / "old.jpg").shape == (72, 144, 3)


def test_a_bad_output_path_is_reported_before_reading_any_face(in_project: Path, capsys) -> None:
    assert main(["5", "6", "wall.txt"]) == 1
    err = capsys.readouterr().err
    assert "face-wall: error: OpenCV cannot save .txt files" in err
    assert "loaded" not in err
