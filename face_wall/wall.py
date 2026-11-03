"""Build a face wall: a grid of face images picked at random from a folder."""

from __future__ import annotations

import logging
import os
import random
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Union

import cv2
import numpy as np

from .faces import FaceCropper

logger = logging.getLogger(__name__)

PathLike = Union[str, os.PathLike[str]]
#: Tile size in pixels: one number for a square tile, or ``(width, height)``.
TileSize = Union[int, tuple[int, int]]

#: Folder the faces are read from when none is given (relative to the working directory).
DEFAULT_FACE_FOLDER = "./faces"
#: Where the wall is saved when no path is given. The extension decides the image format.
DEFAULT_SAVE_PATH = "./wall.jpg"
#: Width and height of one tile in pixels.
DEFAULT_TILE_SIZE = (72, 72)
#: File types picked up from the face folder. Compared case-insensitively, so ``.JPG`` counts too.
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


class FaceWallError(Exception):
    """A problem the user can fix: a bad argument, a missing folder, an unwritable output path."""


def find_images(folder: PathLike, extensions: Iterable[str] = IMAGE_EXTENSIONS) -> list[Path]:
    """List the image files directly inside ``folder``, sorted by name.

    Sorting by name (not by whatever order the file system returns) keeps a seeded wall
    identical on every machine.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FaceWallError(f"face folder not found: {folder}")
    wanted = {ext.lower() for ext in extensions}
    images = sorted(
        (path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in wanted),
        key=lambda path: path.name,
    )
    if not images:
        raise FaceWallError(f"no {', '.join(sorted(wanted))} images in {folder}")
    return images


def read_image(path: PathLike) -> np.ndarray | None:
    """Read an image as a 3-channel BGR array, or return ``None`` if it is not a readable image.

    The file is read as bytes and decoded with ``cv2.imdecode`` because ``cv2.imread`` cannot
    open paths with non-ASCII characters on Windows.
    """
    try:
        data = np.fromfile(os.fspath(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path: PathLike, image: np.ndarray) -> Path:
    """Save ``image`` to ``path``; the extension picks the format (``.jpg``, ``.png``, ...).

    Encodes in memory and writes the bytes ourselves, for the same non-ASCII path reason as
    :func:`read_image`, and so that a failed write raises instead of passing silently.
    """
    path = Path(path)
    if not path.suffix:
        raise FaceWallError(f"cannot tell the image format of {path}: add an extension like .jpg")
    if not path.parent.is_dir():
        raise FaceWallError(f"output folder does not exist: {path.parent}")
    try:
        ok, encoded = cv2.imencode(path.suffix, image)
    except cv2.error:
        ok = False
    if not ok:
        raise FaceWallError(f"OpenCV cannot save {path.suffix} files; use .jpg or .png")
    try:
        encoded.tofile(os.fspath(path))
    except OSError as exc:
        raise FaceWallError(f"cannot save {path}: {exc.strerror or exc}") from exc
    return path


def normalize_tile_size(size: TileSize) -> tuple[int, int]:
    """Turn ``72`` or ``(96, 128)`` into a checked ``(width, height)`` pair."""
    width, height = (size, size) if isinstance(size, int) else size
    for value in (width, height):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise FaceWallError(f"tile size must be whole numbers of at least 1, got {size!r}")
    return width, height


def resize_to_tile(image: np.ndarray, tile_size: tuple[int, int]) -> np.ndarray:
    """Resize ``image`` to exactly ``tile_size`` (the aspect ratio is not kept)."""
    width, height = tile_size
    shrinking = width * height < image.shape[0] * image.shape[1]
    # INTER_AREA avoids aliasing when shrinking; INTER_LINEAR is OpenCV's default otherwise.
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=interpolation)


def load_faces(
    folder: PathLike = DEFAULT_FACE_FOLDER,
    size: TileSize = DEFAULT_TILE_SIZE,
    *,
    extensions: Iterable[str] = IMAGE_EXTENSIONS,
    crop_faces: bool = False,
) -> list[np.ndarray]:
    """Read every image in ``folder`` and resize it to one tile.

    Unreadable files are skipped with a warning. With ``crop_faces`` each image is first
    cropped around its largest face (see :class:`face_wall.faces.FaceCropper`), and images
    without a face are skipped too.
    """
    tile_size = normalize_tile_size(size)
    cropper = FaceCropper() if crop_faces else None
    tiles = []
    for path in find_images(folder, extensions):
        image = read_image(path)
        if image is None:
            logger.warning("skipped %s: not a readable image", path.name)
            continue
        if cropper is not None:
            image = cropper.crop(image, aspect=tile_size[0] / tile_size[1])
            if image is None:
                logger.warning("skipped %s: no face found", path.name)
                continue
        tiles.append(resize_to_tile(image, tile_size))
    if not tiles:
        raise FaceWallError(f"none of the images in {folder} could be used")
    logger.info(
        "loaded %d face%s from %s%s",
        len(tiles),
        "" if len(tiles) == 1 else "s",
        folder,
        " (cropped to the face)" if crop_faces else "",
    )
    return tiles


def _check_count(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise FaceWallError(f"{name} must be a whole number of at least 1, got {value!r}")


def build_wall(
    faces: Sequence[np.ndarray], columns: int, rows: int, seed: int | None = None
) -> np.ndarray:
    """Arrange ``columns`` x ``rows`` tiles, each picked at random from ``faces``.

    Faces may repeat. The same ``seed`` always gives the same wall; ``None`` gives a new one
    every time.
    """
    _check_count("columns", columns)
    _check_count("rows", rows)
    if not faces:
        raise FaceWallError("no faces to build a wall from")
    if len({face.shape for face in faces}) != 1:
        raise FaceWallError("all faces must have the same size; resize them with load_faces()")
    rng = random.Random(seed)
    picks = [rng.randrange(len(faces)) for _ in range(rows * columns)]
    tiles = np.stack(faces)[picks]
    height, width = tiles.shape[1:3]
    grid = tiles.reshape(rows, columns, height, width, -1)
    return grid.transpose(0, 2, 1, 3, 4).reshape(rows * height, columns * width, -1)


def generate_face_wall(
    x: int,
    y: int,
    save_path: PathLike = DEFAULT_SAVE_PATH,
    *,
    faces: PathLike = DEFAULT_FACE_FOLDER,
    size: TileSize = DEFAULT_TILE_SIZE,
    seed: int | None = None,
    crop_faces: bool = False,
) -> Path:
    """Build a wall of ``x`` columns by ``y`` rows from the images in ``faces`` and save it.

    Returns the path the wall was saved to.
    """
    _check_count("columns (x)", x)
    _check_count("rows (y)", y)
    tiles = load_faces(faces, size, crop_faces=crop_faces)
    wall = build_wall(tiles, x, y, seed)
    path = write_image(save_path, wall)
    logger.info("saved a %d x %d wall (%d x %d px) to %s", x, y, wall.shape[1], wall.shape[0], path)
    return path
