"""The dataset checkers' own reading of a dataset directory, written apart from sim/.

``scripts/check_dataset.py`` and ``scripts/compare_datasets.py`` read what ``make_dataset``
wrote without importing the simulator, so a bug in the simulator's own reading of a dataset
cannot hide a bug in its writing. The layout (``sim/dataset.py``) is

  scenario.yaml, setup.json, metadata.jsonl, dataset.json, timing.json, frames/NNNNNN.png

and a sweep root holds one such directory per variant, plus ``variants.json``, the index of
the variants ``make_dataset`` wrote there.

A frame's hash is the sha256 of its shape written as text, followed by its pixels as
little-endian 16-bit numbers. ``pixel_hash`` re-implements that contract; a test holds it equal
to the simulator's. PNGs store RGB frames in OpenCV's BGR order, so ``read_png`` turns them back.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

def read_lines(path: Path) -> list[dict[str, Any]]:
    """Every line of a dataset's metadata.jsonl, parsed."""
    with open(Path(path) / "metadata.jsonl") as fh:
        return [json.loads(line) for line in fh]


def variants(root: Path) -> list[str] | None:
    """The variant directories a sweep root's variants.json lists, or None for a single dataset."""
    index = Path(root) / "variants.json"
    if not index.exists():
        return None
    return [v["dir"] for v in json.loads(index.read_text())["variants"]]


def pixel_hash(frame: np.ndarray) -> str:
    """sha256 of the frame's shape (as text) then its pixels as little-endian 16-bit numbers."""
    digest = hashlib.sha256(str(frame.shape).encode())
    digest.update(np.ascontiguousarray(frame, dtype="<u2").tobytes())
    return digest.hexdigest()


def read_png(path: Path) -> np.ndarray:
    """A stored 16-bit frame: (h, w) mono or (h, w, 3) RGB."""
    frame = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if frame is None:
        raise FileNotFoundError(f"{path}: missing or unreadable")
    return frame if frame.ndim == 2 else frame[..., ::-1]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
