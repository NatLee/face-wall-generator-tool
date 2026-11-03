"""Backward-compatible entry point for running from a clone.

This project used to be this one script: ``python main.py X Y [SAVE_PATH]``, with
``generate_face_wall``, ``get_faces`` and ``FACE_FOLDER`` defined here. The code now lives in
the ``face_wall`` package (installed command: ``face-wall``). This file keeps the old command
and names working; it can go once nobody runs or imports ``main.py`` any more.
"""

from face_wall import DEFAULT_FACE_FOLDER, DEFAULT_TILE_SIZE, IMAGE_EXTENSIONS, load_faces
from face_wall import generate_face_wall as generate_face_wall
from face_wall.cli import main

FACE_FOLDER = DEFAULT_FACE_FOLDER


def get_faces(folder_path=FACE_FOLDER, extensions=IMAGE_EXTENSIONS, size=DEFAULT_TILE_SIZE):
    """Old name for :func:`face_wall.load_faces`."""
    return load_faces(folder_path, size, extensions=extensions)


if __name__ == "__main__":
    raise SystemExit(main())
