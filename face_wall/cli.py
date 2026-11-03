"""The ``face-wall X Y [SAVE_PATH]`` command (also ``main.py`` and ``python -m face_wall``)."""

from __future__ import annotations

import argparse
import contextlib
import logging
import sys
from collections.abc import Iterator, Sequence

from . import __version__
from .wall import (
    DEFAULT_FACE_FOLDER,
    DEFAULT_SAVE_PATH,
    DEFAULT_TILE_SIZE,
    IMAGE_EXTENSIONS,
    FaceWallError,
    generate_face_wall,
)

PROG = "face-wall"

DESCRIPTION = """\
Tile face images into an X-by-Y grid (a "face wall"). Each cell is a face picked
at random from a folder, so you get a test image with a known number of faces in
known places - handy for checking face detection and recognition models.
"""

EXAMPLES = f"""\
examples:
  {PROG} 8 6
      8 columns x 6 rows from ./faces, saved as ./wall.jpg
  {PROG} 8 6 out/wall.png --seed 42
      the same wall every time, saved as PNG in ./out (the folder must exist)
  {PROG} 12 4 --size 96x128
      taller tiles (width x height)
  {PROG} 6 2 --faces photos --crop-faces
      crop each photo around its face first

The old python-fire spellings still work: -x 8 -y 6 -save_path wall.jpg, --x=8, --save-path.
"""


def _whole_number(text: str) -> int:
    try:
        return int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number, got {text!r}") from None


def _tile_size(text: str) -> tuple[int, int]:
    width, sep, height = text.lower().partition("x")
    try:
        return (int(width), int(height)) if sep else (int(width), int(width))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected N or WIDTHxHEIGHT, got {text!r}") from None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        usage="%(prog)s [options] X Y [SAVE_PATH]",
        description=DESCRIPTION,
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    # X, Y and SAVE_PATH can be given by position or as flags, in any order, like the original
    # python-fire command line; _grid_arguments() sorts them out.
    parser.add_argument("pos1", nargs="?", metavar="X", help="number of columns")
    parser.add_argument("pos2", nargs="?", metavar="Y", help="number of rows")
    parser.add_argument(
        "pos3",
        nargs="?",
        metavar="SAVE_PATH",
        help=f"output image; the extension picks the format (default: {DEFAULT_SAVE_PATH})",
    )
    parser.add_argument("-x", "--x", type=_whole_number, metavar="X", help="same as X")
    parser.add_argument("-y", "--y", type=_whole_number, metavar="Y", help="same as Y")
    parser.add_argument(
        "--save-path",
        "--save_path",
        "-save_path",
        dest="save_path",
        metavar="PATH",
        help="same as SAVE_PATH",
    )
    parser.add_argument(
        "--faces",
        default=DEFAULT_FACE_FOLDER,
        metavar="DIR",
        help=f"folder with {'/'.join(IMAGE_EXTENSIONS)} images (default: {DEFAULT_FACE_FOLDER})",
    )
    parser.add_argument(
        "--size",
        type=_tile_size,
        default=DEFAULT_TILE_SIZE,
        metavar="PX",
        help=f"tile size: N for N x N, or WIDTHxHEIGHT (default: {DEFAULT_TILE_SIZE[0]})",
    )
    parser.add_argument(
        "--seed",
        type=_whole_number,
        metavar="N",
        help="random seed; the same seed and faces always give the same wall (default: random)",
    )
    parser.add_argument(
        "--crop-faces",
        action="store_true",
        help="find the largest face in each image (YuNet detector) and crop around it; "
        "images without a face are skipped",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _grid_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> tuple[int, int, str]:
    """Combine flags and positional values the way python-fire did.

    Flags fill their parameter first; positional values then fill the parameters that are
    still empty, in the order X, Y, SAVE_PATH. So ``out.jpg -x 3 -y 2`` saves to ``out.jpg``.
    """
    values = {"x": args.x, "y": args.y, "save_path": args.save_path}
    positionals = [value for value in (args.pos1, args.pos2, args.pos3) if value is not None]
    for name, value in values.items():
        if value is None and positionals:
            values[name] = positionals.pop(0)
    if positionals:
        parser.error(f"unexpected extra argument: {positionals[0]}")
    for name, meaning in (("x", "columns"), ("y", "rows")):
        if values[name] is None:
            parser.error(f"missing {name.upper()} (number of {meaning})")
        if isinstance(values[name], str):
            try:
                values[name] = _whole_number(values[name])
            except argparse.ArgumentTypeError as exc:
                parser.error(f"{name.upper()}: {exc}")
    return values["x"], values["y"], values["save_path"] or DEFAULT_SAVE_PATH


class _CliFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        level = f"{record.levelname.lower()}: " if record.levelno >= logging.WARNING else ""
        return f"{PROG}: {level}{record.getMessage()}"


@contextlib.contextmanager
def _log_to_stderr() -> Iterator[None]:
    """Show the library's progress messages and warnings on stderr while the command runs."""
    logger = logging.getLogger(__package__)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_CliFormatter())
    previous = logger.level, logger.propagate
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        yield
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous[0])
        logger.propagate = previous[1]


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_intermixed_args(argv)
    x, y, save_path = _grid_arguments(parser, args)
    with _log_to_stderr():
        try:
            generate_face_wall(
                x,
                y,
                save_path,
                faces=args.faces,
                size=args.size,
                seed=args.seed,
                crop_faces=args.crop_faces,
            )
        except FaceWallError as exc:
            logging.getLogger(__package__).error("%s", exc)
            return 1
    return 0
