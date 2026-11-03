from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from face_wall import write_image

REPO_ROOT = Path(__file__).resolve().parent.parent
#: Real portraits (public domain, see CREDITS.md) used to check the face detector.
PORTRAITS = REPO_ROOT / "examples" / "portraits"

#: Three easy-to-tell-apart BGR colours, one per fake "face".
COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]


def solid(color, width=40, height=30) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = color
    return image


@pytest.fixture
def face_dir(tmp_path: Path) -> Path:
    """A folder with three solid-colour PNG "faces" of different sizes."""
    folder = tmp_path / "faces"
    folder.mkdir()
    for index, color in enumerate(COLORS):
        write_image(folder / f"face{index}.png", solid(color, 40 + index * 10, 30 + index * 5))
    return folder
