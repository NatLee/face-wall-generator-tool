"""Find the main face in a photo and crop around it.

Uses YuNet, a small face detector from the OpenCV model zoo that runs through OpenCV's own
``cv2.FaceDetectorYN``. The model file (about 230 KB, MIT licence) ships inside this package,
so no download and no extra dependency is needed.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from importlib import resources

import cv2
import numpy as np

#: YuNet export with dynamic input size. It runs on OpenCV 4.x and on the ONNX Runtime engine
#: of OpenCV 5.x (the older fixed-size ``2023mar`` export does not). Licence: models/LICENSE-YuNet.
MODEL_FILE = "face_detection_yunet_2026may.onnx"
#: YuNet finds faces between about 10 and 300 px. Shrinking the longer side of the photo to
#: this many pixels before detecting keeps a portrait's face in that range and keeps it fast.
DETECT_MAX_SIDE = 320
#: Detections with a lower confidence (0 to 1) are ignored.
SCORE_THRESHOLD = 0.7
#: The crop is this many times larger than the detected face box, so a tile shows the whole
#: head (hair, chin) instead of only the eyes-to-mouth area YuNet marks.
CROP_SCALE = 1.6

#: ``(x, y, width, height)`` of a face in image pixels.
Box = tuple[float, float, float, float]


@contextlib.contextmanager
def _quiet_opencv() -> Iterator[None]:
    """Hide OpenCV's own warnings for the duration of the block.

    OpenCV 5 prints "Targets are not supported by the new graph engine" whenever a detector is
    created, which is harmless but looks like a problem in a command-line tool. Uses the public
    ``cv2.utils.logging`` API when the installed OpenCV has it, and does nothing otherwise.
    """
    log = getattr(getattr(cv2, "utils", None), "logging", None)
    if log is None:
        yield
        return
    previous = log.getLogLevel()
    log.setLogLevel(log.LOG_LEVEL_ERROR)
    try:
        yield
    finally:
        log.setLogLevel(previous)


def _plausible_boxes(boxes: np.ndarray, width: int, height: int) -> list[Box]:
    """Keep the boxes that are finite, have a size and are centred inside the image.

    On OpenCV 4.x the detector sometimes returns junk rows (``inf`` coordinates) for inputs of
    32 px or less, so its output is checked before use.
    """
    plausible = []
    for x, y, w, h in boxes.tolist():
        if not all(np.isfinite((x, y, w, h))) or w < 1 or h < 1:
            continue
        if 0 <= x + w / 2 <= width and 0 <= y + h / 2 <= height:
            plausible.append((x, y, w, h))
    return plausible


def crop_window(
    box: Box,
    image_size: tuple[int, int],
    aspect: float = 1.0,
    scale: float = CROP_SCALE,
) -> tuple[slice, slice]:
    """Pick the crop around a face box: rows and columns to slice the image with.

    The crop has the requested ``aspect`` (width / height), is ``scale`` times the face box,
    stays inside the image (it is shifted, and shrunk only if the image is too small) and is
    centred on the face when it can be.
    """
    x, y, width, height = box
    image_width, image_height = image_size
    crop_height = max(height, width / aspect) * scale
    crop_width = crop_height * aspect
    fit = min(1.0, image_width / crop_width, image_height / crop_height)
    crop_width, crop_height = crop_width * fit, crop_height * fit
    left = min(max(x + width / 2 - crop_width / 2, 0.0), image_width - crop_width)
    top = min(max(y + height / 2 - crop_height / 2, 0.0), image_height - crop_height)
    rows = slice(round(top), max(round(top + crop_height), round(top) + 1))
    columns = slice(round(left), max(round(left + crop_width), round(left) + 1))
    return rows, columns


class FaceCropper:
    """Detect the largest face in an image and crop the image around it."""

    def __init__(self, score_threshold: float = SCORE_THRESHOLD) -> None:
        model = (resources.files(__package__) / "models" / MODEL_FILE).read_bytes()
        # Loading the model from memory avoids OpenCV's trouble with non-ASCII paths on Windows.
        with _quiet_opencv():
            self._detector = cv2.FaceDetectorYN.create(
                "onnx",
                np.frombuffer(model, dtype=np.uint8),
                np.empty(0, dtype=np.uint8),
                (DETECT_MAX_SIDE, DETECT_MAX_SIDE),
                score_threshold,
            )

    def find_face(self, image: np.ndarray) -> Box | None:
        """Return the box of the largest face in ``image``, or ``None`` if there is none."""
        height, width = image.shape[:2]
        scale = min(1.0, DETECT_MAX_SIDE / max(height, width))
        small = image
        if scale < 1.0:
            size = (max(1, round(width * scale)), max(1, round(height * scale)))
            small = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        self._detector.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self._detector.detect(small)
        if faces is None:
            return None
        boxes = _plausible_boxes(faces[:, :4], small.shape[1], small.shape[0])
        if not boxes:
            return None
        x, y, w, h = max(boxes, key=lambda box: box[2] * box[3])
        return x / scale, y / scale, w / scale, h / scale

    def crop(self, image: np.ndarray, aspect: float = 1.0) -> np.ndarray | None:
        """Crop ``image`` around its largest face, or return ``None`` if no face is found."""
        box = self.find_face(image)
        if box is None:
            return None
        rows, columns = crop_window(box, (image.shape[1], image.shape[0]), aspect)
        return image[rows, columns]
