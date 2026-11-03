from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
from conftest import COLORS, solid

from face_wall import (
    DEFAULT_TILE_SIZE,
    FaceWallError,
    build_wall,
    find_images,
    generate_face_wall,
    load_faces,
    read_image,
    write_image,
)
from face_wall.wall import normalize_tile_size


def test_find_images_ignores_extension_case(tmp_path: Path) -> None:
    # Regression: only lower-case .png/.jpg were picked up, so A.JPG and b.jpeg were ignored.
    for name in ["a.png", "B.JPG", "c.jpeg", "d.Png", "notes.txt"]:
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "folder.png").mkdir()
    assert [p.name for p in find_images(tmp_path)] == ["B.JPG", "a.png", "c.jpeg", "d.Png"]


def test_find_images_reports_missing_folder(tmp_path: Path) -> None:
    with pytest.raises(FaceWallError, match="face folder not found"):
        find_images(tmp_path / "nope")


def test_find_images_reports_empty_folder(tmp_path: Path) -> None:
    # Regression: an empty folder ended in "IndexError: Cannot choose from an empty sequence".
    with pytest.raises(FaceWallError, match=r"no \.jpeg, \.jpg, \.png images"):
        find_images(tmp_path)


def test_load_faces_resizes_every_image_to_one_tile(face_dir: Path) -> None:
    tiles = load_faces(face_dir)
    assert len(tiles) == len(COLORS)
    assert {tile.shape for tile in tiles} == {(DEFAULT_TILE_SIZE[1], DEFAULT_TILE_SIZE[0], 3)}
    assert {tile.shape for tile in load_faces(face_dir, (30, 40))} == {(40, 30, 3)}


def test_load_faces_skips_unreadable_files(face_dir: Path, caplog) -> None:
    # Regression: a broken image crashed cv2.resize with an OpenCV assertion.
    (face_dir / "broken.jpg").write_bytes(b"not an image")
    (face_dir / "empty.png").write_bytes(b"")
    with caplog.at_level(logging.WARNING, logger="face_wall"):
        tiles = load_faces(face_dir)
    assert len(tiles) == len(COLORS)
    assert "skipped broken.jpg: not a readable image" in caplog.text
    assert "skipped empty.png: not a readable image" in caplog.text


def test_load_faces_fails_when_nothing_is_usable(tmp_path: Path) -> None:
    (tmp_path / "broken.jpg").write_bytes(b"not an image")
    with pytest.raises(FaceWallError, match="none of the images"):
        load_faces(tmp_path)


def test_build_wall_has_columns_across_and_rows_down() -> None:
    faces = [solid(color, 10, 8) for color in COLORS]
    wall = build_wall(faces, columns=4, rows=3, seed=1)
    assert wall.shape == (3 * 8, 4 * 10, 3)
    for row in range(3):
        for column in range(4):
            tile = wall[row * 8 : (row + 1) * 8, column * 10 : (column + 1) * 10]
            assert any(np.array_equal(tile, face) for face in faces)


def test_build_wall_is_reproducible_with_a_seed() -> None:
    faces = [solid(color, 4, 4) for color in COLORS]
    first = build_wall(faces, 10, 10, seed=42)
    assert np.array_equal(first, build_wall(faces, 10, 10, seed=42))
    assert not np.array_equal(first, build_wall(faces, 10, 10, seed=43))


@pytest.mark.parametrize("count", [0, -1, 2.5, True, "3"])
def test_build_wall_rejects_bad_counts(count) -> None:
    # Regression: 0 columns ended in "ValueError: need at least one array to concatenate".
    with pytest.raises(FaceWallError, match="must be a whole number of at least 1"):
        build_wall([solid(COLORS[0])], count, 2)


def test_build_wall_rejects_faces_of_different_sizes() -> None:
    with pytest.raises(FaceWallError, match="same size"):
        build_wall([solid(COLORS[0], 4, 4), solid(COLORS[1], 5, 4)], 2, 2)


@pytest.mark.parametrize("size, expected", [(72, (72, 72)), ((96, 128), (96, 128))])
def test_normalize_tile_size(size, expected) -> None:
    assert normalize_tile_size(size) == expected


@pytest.mark.parametrize("size", [0, (10, 0), (10, 2.5), (True, 3)])
def test_normalize_tile_size_rejects_bad_values(size) -> None:
    with pytest.raises(FaceWallError, match="tile size"):
        normalize_tile_size(size)


def test_images_round_trip_through_non_ascii_paths(tmp_path: Path) -> None:
    # Regression: cv2.imread/imwrite cannot handle non-ASCII paths on Windows (CI runs there).
    folder = tmp_path / "臉牆 faces"
    folder.mkdir()
    image = solid(COLORS[1], 12, 9)
    write_image(folder / "人像.png", image)
    assert np.array_equal(read_image(folder / "人像.png"), image)
    assert [tile.shape for tile in load_faces(folder, 6)] == [(6, 6, 3)]


def test_write_image_reports_missing_folder(tmp_path: Path) -> None:
    # Regression: the old script printed nothing, saved nothing and exited with 0.
    with pytest.raises(FaceWallError, match="output folder does not exist"):
        write_image(tmp_path / "missing" / "wall.jpg", solid(COLORS[0]))


@pytest.mark.parametrize("name, message", [("wall.txt", "cannot save .txt"), ("wall", "extension")])
def test_write_image_reports_unusable_formats(tmp_path: Path, name: str, message: str) -> None:
    with pytest.raises(FaceWallError, match=message):
        write_image(tmp_path / name, solid(COLORS[0]))


def test_generate_face_wall_saves_to_wall_jpg_in_the_working_directory(
    face_dir: Path, monkeypatch
) -> None:
    monkeypatch.chdir(face_dir.parent)
    path = generate_face_wall(2, 3)
    assert path == Path("./wall.jpg")
    assert read_image(face_dir.parent / "wall.jpg").shape == (3 * 72, 2 * 72, 3)


def test_generate_face_wall_options(face_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "wall.png"
    generate_face_wall(5, 2, out, faces=face_dir, size=(20, 30), seed=7)
    first = read_image(out)
    assert first.shape == (2 * 30, 5 * 20, 3)
    generate_face_wall(5, 2, out, faces=face_dir, size=(20, 30), seed=7)
    assert np.array_equal(read_image(out), first)
