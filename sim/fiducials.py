"""Fiducial markers: printed ArUco squares on the bezel that let the camera find the screen.

Why markers. The detector must know which camera pixel looks at which screen millimetre (a
homography) without projecting anything. Printed markers on the bezel are passive: they need
only the room light, or a small lamp of their own, and they work whatever content is showing.
Each is an ArUco code from the DICT_4X4_50 dictionary: a 6 x 6 grid of square cells, 4 x 4
data bits inside a one-cell black border, printed on white paper. A one-cell white quiet zone
around the black square gives the detector the contrast ring it needs to find the square.

Layout. Eight markers, ids 0-7. One sits in each bezel corner (0 TL, 1 TR, 2 BR, 3 BL), so a
camera that sees the whole screen sees all four corners of the frame. Two more sit on each
side of the overlap (4, 5 above and 6, 7 below a vertical overlap; 4, 5 left and 6, 7 right of
a horizontal one), so a camera zoomed on the overlap still sees at least four.

Rendering. Paper and ink are painted into the screen grid's reflectance map by exact area
coverage, *per material*. The ink coverage of every grid pixel near a marker is
C_y^T M C_x, where M is the marker's 6 x 6 cell matrix (1 = black) and C_x (C_y) holds how much
of each grid column (row) each cell column (row) covers. Painting cell by cell instead would
blend each cell over the last and get pixels shared by two black cells wrong (by up to a
quarter of the paper-ink contrast), which would bias the marker edges the detector measures.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import cv2
import numpy as np

from sim.cfg import check_keys, choice, integer, num, pair
from sim.planar import apply_h, rect_polygon
from sim.screen import Screen, ScreenGrid

DICTIONARY = "DICT_4X4_50"
CELLS = 6  # 4 x 4 data bits inside a one-cell black border
GAP_MM = 10.0  # least gap between two markers' paper, and between paper and the bezel's edges


@lru_cache(maxsize=None)
def cell_matrix(marker_id: int) -> np.ndarray:
    """(6, 6) ink matrix of DICT_4X4_50 marker `marker_id`: 1 = black cell, row 0 at the top."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    image = cv2.aruco.generateImageMarker(dictionary, int(marker_id), CELLS)
    cells = (image == 0).astype(np.float64)
    cells.setflags(write=False)
    return cells


@dataclass(frozen=True)
class MarkerSet:
    """Printed markers, upright on the bezel: marker id i is centred at centres_mm[i]."""

    centres_mm: tuple[tuple[float, float], ...]
    size_mm: float = 80.0  # side of the black square (6 cells)
    quiet_zone_cells: int = 1  # white paper around the square, in cells
    paper: float = 0.8  # reflectance of the white paper
    ink: float = 0.04  # reflectance of the black ink

    def __post_init__(self) -> None:
        if not 0 < len(self.centres_mm) <= 50:
            raise ValueError("markers: need 1 to 50 markers (DICT_4X4_50)")
        if self.size_mm <= 0 or self.quiet_zone_cells < 1:
            raise ValueError("markers: need size_mm > 0 and quiet_zone_cells >= 1 (detection needs the quiet zone)")
        if not 0 < self.ink < self.paper <= 1:
            raise ValueError("markers: need 0 < ink < paper <= 1")

    @property
    def cell_mm(self) -> float:
        return self.size_mm / CELLS

    @property
    def footprint_mm(self) -> float:
        """Side of the printed paper: the square plus its quiet zone on both sides."""
        return self.size_mm + 2 * self.quiet_zone_cells * self.cell_mm

    def footprint(self, marker_id: int) -> tuple[float, float, float, float]:
        cx, cy = self.centres_mm[marker_id]
        h = self.footprint_mm / 2
        return cx - h, cy - h, cx + h, cy + h

    def square(self, marker_id: int) -> np.ndarray:
        """Corners (TL, TR, BR, BL) of the black square in screen mm, ArUco's corner order."""
        cx, cy = self.centres_mm[marker_id]
        h = self.size_mm / 2
        return rect_polygon(cx - h, cy - h, cx + h, cy + h)

    def to_setup_dict(self) -> dict[str, Any]:
        """The marker layout as measured at install: what the detector may read."""
        return {
            "dictionary": DICTIONARY,
            "size_mm": float(self.size_mm),
            "quiet_zone_cells": int(self.quiet_zone_cells),
            "items": [
                {"id": i, "centre_mm": [float(x), float(y)], "rotation_deg": 0.0}
                for i, (x, y) in enumerate(self.centres_mm)
            ],
        }


def auto_layout(
    screen: Screen, overlap: np.ndarray, size_mm: float, quiet_zone_cells: int
) -> tuple[tuple[float, float], ...]:
    """Centres of the 8 standard markers: bezel corners, then two on each side of the overlap.

    For a vertical overlap (taller than wide) the extra markers sit on the top and bottom bezel,
    above and below the overlap's left and right edges; for a horizontal one, on the left and
    right bezel beside its top and bottom edges. They are kept clear of the corner markers.
    """
    b = screen.bezel.width_mm
    fp = size_mm * (CELLS + 2 * quiet_zone_cells) / CELLS
    if fp + 2 * GAP_MM > b:
        raise ValueError(f"markers: a {fp:.1f} mm marker needs a bezel of at least {fp + 2 * GAP_MM:.1f} mm, got {b}")
    w, h = screen.size_mm
    m = b / 2  # the bezel's centre line
    centres = [(-m, -m), (w + m, -m), (w + m, h + m), (-m, h + m)]
    lo, hi = np.min(overlap, axis=0), np.max(overlap, axis=0)
    vertical = (hi[1] - lo[1]) >= (hi[0] - lo[0])
    axis = 0 if vertical else 1  # the screen axis along which the pair is spread
    length = w if vertical else h
    middle = (lo[axis] + hi[axis]) / 2
    half = max((hi[axis] - lo[axis]) / 2, (fp + GAP_MM) / 2)
    first = max(middle - half, -m + fp + GAP_MM)  # clear of the corner markers
    second = min(middle + half, length + m - fp - GAP_MM)
    if second - first < fp + GAP_MM:
        raise ValueError("markers: no room for two markers beside the overlap")
    if vertical:
        centres += [(first, -m), (second, -m), (first, h + m), (second, h + m)]
    else:
        centres += [(-m, first), (-m, second), (w + m, first), (w + m, second)]
    return tuple((float(x), float(y)) for x, y in centres)


def check_on_bezel(markers: MarkerSet, screen: Screen) -> None:
    """Every marker's paper must lie on the bezel, off the screen, and clear of the others."""
    x0, y0, x1, y1 = screen.extent_mm
    w, h = screen.size_mm
    feet = [markers.footprint(i) for i in range(len(markers.centres_mm))]
    for i, (a0, b0, a1, b1) in enumerate(feet):
        if a0 < x0 or b0 < y0 or a1 > x1 or b1 > y1:
            raise ValueError(f"markers: marker {i} does not fit on the bezel")
        if a0 < w and a1 > 0 and b0 < h and b1 > 0:
            raise ValueError(f"markers: marker {i} lies on the screen, not on the bezel")
        for j in range(i):
            c0, d0, c1, d1 = feet[j]
            if a0 < c1 and c0 < a1 and b0 < d1 and d0 < b1:
                raise ValueError(f"markers: markers {j} and {i} overlap")


def paint(refl: np.ndarray, grid: ScreenGrid, markers: MarkerSet, background: float) -> None:
    """Paint paper and ink into reflectance map `refl` (float32, on `grid`), in place.

    `background` is the reflectance the paper covers (the bezel's): exact as long as each
    marker's paper lies wholly on the bezel, which :func:`check_on_bezel` guarantees.
    """
    cell = markers.cell_mm
    for marker_id in range(len(markers.centres_mm)):
        rs, cs, foot = grid.rect_coverage(markers.footprint(marker_id))
        sx, sy = markers.square(marker_id)[0]
        cx = np.stack([grid.coverage(sx + k * cell, sx + (k + 1) * cell, 0, cs.start, cs.stop) for k in range(CELLS)])
        cy = np.stack([grid.coverage(sy + k * cell, sy + (k + 1) * cell, 1, rs.start, rs.stop) for k in range(CELLS)])
        ink = (cy.T @ cell_matrix(marker_id) @ cx).astype(np.float32)
        refl[rs, cs] += np.float32(markers.paper - background) * foot + np.float32(markers.ink - markers.paper) * ink


def visible(markers: MarkerSet, h_mm_to_px: np.ndarray, resolution: tuple[int, int], margin_px: float = 2.0) -> list[int]:
    """Ids of markers whose whole paper lies inside the camera frame, `margin_px` from its edges."""
    w, h = resolution
    out = []
    for i in range(len(markers.centres_mm)):
        corners = apply_h(h_mm_to_px, rect_polygon(*markers.footprint(i)))
        lo, hi = corners.min(axis=0), corners.max(axis=0)
        inside_lo = lo[0] >= margin_px - 0.5 and lo[1] >= margin_px - 0.5
        inside_hi = hi[0] <= w - 0.5 - margin_px and hi[1] <= h - 0.5 - margin_px
        if inside_lo and inside_hi:
            out.append(i)
    return out


_KEYS = {"layout", "centres_mm", "size_mm", "quiet_zone_cells", "paper", "ink"}


def from_config(cfg: Mapping[str, Any], screen: Screen, overlap: np.ndarray) -> MarkerSet:
    """Build the marker set from a scenario's ``screen.bezel.markers`` block."""
    check_keys(cfg, _KEYS, "screen.bezel.markers")
    size = num(cfg.get("size_mm", 80.0), "markers.size_mm")
    quiet = integer(cfg.get("quiet_zone_cells", 1), "markers.quiet_zone_cells")
    layout = choice(cfg.get("layout", "auto"), ("auto", "explicit"), "markers.layout")
    if layout == "explicit":
        if "centres_mm" not in cfg:
            raise ValueError("markers: layout explicit needs centres_mm")
        centres = tuple(pair(c, "markers.centres_mm") for c in cfg["centres_mm"])
    else:
        if "centres_mm" in cfg:
            raise ValueError("markers: centres_mm is only used with layout: explicit")
        centres = auto_layout(screen, overlap, size, quiet)
    markers = MarkerSet(
        centres_mm=centres,
        size_mm=size,
        quiet_zone_cells=quiet,
        paper=num(cfg.get("paper", 0.8), "markers.paper"),
        ink=num(cfg.get("ink", 0.04), "markers.ink"),
    )
    check_on_bezel(markers, screen)
    return markers
