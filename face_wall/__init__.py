"""Tile face images into a grid (a "face wall") to test face detection and recognition models."""

from .wall import (
    DEFAULT_FACE_FOLDER,
    DEFAULT_SAVE_PATH,
    DEFAULT_TILE_SIZE,
    IMAGE_EXTENSIONS,
    FaceWallError,
    build_wall,
    find_images,
    generate_face_wall,
    load_faces,
    read_image,
    write_image,
)

__version__ = "1.0.0"

__all__ = [
    "DEFAULT_FACE_FOLDER",
    "DEFAULT_SAVE_PATH",
    "DEFAULT_TILE_SIZE",
    "IMAGE_EXTENSIONS",
    "FaceWallError",
    "__version__",
    "build_wall",
    "find_images",
    "generate_face_wall",
    "load_faces",
    "read_image",
    "write_image",
]
