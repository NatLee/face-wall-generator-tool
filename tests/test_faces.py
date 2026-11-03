from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np
import pytest
from conftest import PORTRAITS

from face_wall import load_faces, read_image, write_image
from face_wall.cli import main
from face_wall.faces import CROP_SCALE, FaceCropper, _plausible_boxes, crop_window

PORTRAIT_FILES = sorted(PORTRAITS.glob("*.jpg"))


@pytest.fixture(scope="module")
def cropper() -> FaceCropper:
    return FaceCropper()


def test_crop_window_is_centred_on_the_face() -> None:
    rows, columns = crop_window((40, 40, 20, 20), (100, 100))
    side = 20 * CROP_SCALE
    assert (rows.start, rows.stop) == (round(50 - side / 2), round(50 + side / 2))
    assert (columns.start, columns.stop) == (rows.start, rows.stop)


def test_crop_window_shifts_to_stay_inside_the_image() -> None:
    rows, columns = crop_window((0, 80, 20, 20), (100, 100))
    assert columns.start == 0
    assert rows.stop == 100
    assert rows.stop - rows.start == columns.stop - columns.start == round(20 * CROP_SCALE)


def test_crop_window_keeps_the_aspect_ratio_and_shrinks_to_fit() -> None:
    rows, columns = crop_window((10, 10, 80, 30), (100, 60), aspect=2.0)
    height, width = rows.stop - rows.start, columns.stop - columns.start
    assert (width, height) == (100, 50)
    assert rows.start >= 0 and rows.stop <= 60


def test_model_loads_without_opencv_noise(capfd) -> None:
    # OpenCV 5 logs "Targets are not supported by the new graph engine" when the detector is
    # created; the command line should not show that to users.
    FaceCropper()
    assert "graph engine" not in capfd.readouterr().err


@pytest.mark.parametrize("path", PORTRAIT_FILES, ids=lambda path: path.stem)
def test_finds_and_crops_the_face_in_each_example_portrait(cropper, path: Path) -> None:
    image = read_image(path)
    box = cropper.find_face(image)
    assert box is not None
    crop = cropper.crop(image)
    height, width = crop.shape[:2]
    assert abs(width - height) <= 1
    # The face fills a good part of the crop: CROP_SCALE times its box, unless the image
    # edge forced a smaller crop.
    assert max(box[2], box[3]) <= max(width, height) <= max(box[2], box[3]) * CROP_SCALE + 2


def test_crop_follows_the_requested_aspect_ratio(cropper) -> None:
    crop = cropper.crop(read_image(PORTRAIT_FILES[0]), aspect=0.75)
    height, width = crop.shape[:2]
    assert width / height == pytest.approx(0.75, abs=0.01)


def test_junk_detections_are_ignored() -> None:
    # Regression: OpenCV 4.x sometimes returned rows like (-inf, -1e20, inf, 0) for tiny inputs,
    # which made a 5 x 5 black image "contain" a face now and then.
    rows = np.array(
        [
            [-np.inf, -1.5e20, np.inf, 0.0],  # junk seen from OpenCV 4.10
            [10, 10, 0, 20],  # no width
            [500, 10, 20, 20],  # centred outside a 100 x 100 image
            [20, 30, 40, 50],  # a real-looking box
        ],
        dtype=np.float32,
    )
    assert _plausible_boxes(rows, 100, 100) == [(20, 30, 40, 50)]


def test_no_face_gives_none(cropper) -> None:
    assert cropper.find_face(np.full((240, 320, 3), 128, dtype=np.uint8)) is None
    assert cropper.crop(np.zeros((5, 5, 3), dtype=np.uint8)) is None


def test_load_faces_with_crop_faces_skips_images_without_a_face(tmp_path: Path, caplog) -> None:
    shutil.copy(PORTRAIT_FILES[0], tmp_path / PORTRAIT_FILES[0].name)
    write_image(tmp_path / "wall-paint.png", np.full((200, 200, 3), 200, dtype=np.uint8))
    with caplog.at_level(logging.INFO, logger="face_wall"):
        tiles = load_faces(tmp_path, 64, crop_faces=True)
    assert [tile.shape for tile in tiles] == [(64, 64, 3)]
    assert "skipped wall-paint.png: no face found" in caplog.text
    assert "loaded 1 face from" in caplog.text
    assert "(cropped to the face)" in caplog.text


def test_crop_faces_from_the_command_line(tmp_path: Path, capsys) -> None:
    out = tmp_path / "portraits.jpg"
    argv = ["4", "1", str(out), "--faces", str(PORTRAITS), "--crop-faces", "--seed", "1"]
    assert main(argv) == 0
    assert read_image(out).shape == (72, 4 * 72, 3)
    assert f"loaded {len(PORTRAIT_FILES)} faces from" in capsys.readouterr().err
