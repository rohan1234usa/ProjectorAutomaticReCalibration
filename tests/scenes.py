"""Small installations for tests: same physics and pixel pitch as the demo scene, ~1/8 the size."""

from __future__ import annotations

import cv2
import numpy as np

from sim.calibration import CalibrationSetup
from sim.camera import Camera
from sim.planar import (
    apply_h,
    boundary_distance,
    clip_convex,
    homography_from_points,
    points_in_convex,
    raster_corners,
    rect_polygon,
)
from sim.projector import Projector, side_by_side
from sim.render import QUALITY, Renderer
from sim.screen import Screen, ScreenGrid

RES = (240, 135)  # projector pixels
WIDTH_MM = 250.0  # image width -> 1.042 mm per pixel, like the demo scene
SCREEN_MM = (600.0, 260.0)
CAMERA_RES = (480, 208)


def make_setup(corners: dict[str, np.ndarray], content_rect, shape: str = "cosine", res=RES) -> CalibrationSetup:
    h = {n: homography_from_points(raster_corners(res), np.asarray(c, dtype=float)) for n, c in corners.items()}
    return CalibrationSetup(h_cal=h, resolution={n: res for n in corners}, content_rect_mm=content_rect, blend_shape=shape)


def side_by_side_setup(overlap_mm: float = 60.0, vertical_offset_mm: float = 0.0, shape: str = "cosine") -> CalibrationSetup:
    a, b, rect = side_by_side(SCREEN_MM, RES, RES, WIDTH_MM, WIDTH_MM, overlap_mm, vertical_offset_mm)
    return make_setup({"a": a, "b": b}, rect, shape)


def _box(x: float, y: float, w: float = WIDTH_MM, h: float = WIDTH_MM * RES[1] / RES[0]) -> np.ndarray:
    return rect_polygon(x, y, x + w, y + h)


def rotated_setup(angle_deg: float = 5.0) -> CalibrationSetup:
    """B rotated by `angle_deg` about its centre, overlapping A by about 60 mm."""
    a = _box(80.0, 60.0)
    b = _box(270.0, 60.0)
    c = b.mean(axis=0)
    t = np.radians(angle_deg)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    b = (b - c) @ rot.T + c
    return make_setup({"a": a, "b": b}, _bbox(a, b))


def corner_setup() -> CalibrationSetup:
    """A top-left, B bottom-right: the two meet at a corner overlap of 60 x 50 mm."""
    a = _box(80.0, 30.0)
    b = _box(270.0, 120.625)
    return make_setup({"a": a, "b": b}, _bbox(a, b))


def _bbox(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float]:
    pts = np.vstack([a, b])
    return float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())


def make_renderer(
    setup: CalibrationSetup,
    quality: str = "fast",
    black_level: float = 1.0 / 1500.0,
    ambient: float = 0.0003,
    camera_res: tuple[int, int] = CAMERA_RES,
) -> Renderer:
    projectors = {n: Projector(name=n, resolution=setup.resolution[n], black_level=black_level) for n in setup.names}
    screen = Screen(size_mm=SCREEN_MM, ambient=ambient)
    camera = Camera.whole_screen(SCREEN_MM, camera_res)
    return Renderer(screen, projectors, setup, camera, QUALITY[quality])


def grid_masks(setup: CalibrationSetup, grid: ScreenGrid, margin_px: float) -> dict[str, np.ndarray]:
    """Boolean masks on the screen grid, eroded by `margin_px` grid pixels.

    Keys: union (covered by either projector, inside the content), overlap, only_<name>.
    """
    pts = grid.centers_mm()
    a, b = setup.names
    in_a = points_in_convex(setup.box_mm(a), pts)
    in_b = points_in_convex(setup.box_mm(b), pts)
    in_c = points_in_convex(rect_polygon(*setup.content_rect_mm), pts)
    radius = max(1, int(np.ceil(margin_px)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))

    def erode(m: np.ndarray) -> np.ndarray:
        return cv2.erode(m.astype(np.uint8), kernel).astype(bool)

    union = erode((in_a | in_b) & in_c)
    return {
        "union": union,
        "overlap": union & erode(in_a & in_b),
        f"only_{a}": union & erode(in_a & ~in_b),
        f"only_{b}": union & erode(in_b & ~in_a),
    }


def notch_points(setup: CalibrationSetup) -> list[np.ndarray]:
    """Overlap corners lying on both boxes' outlines: where an inner edge meets the image outline.

    At such a point the blend must go from 0 to 1 in an arbitrarily small distance, so no blend
    can be reproduced exactly by the projectors' pixel blur nearby.
    """
    a, b = setup.names
    box_a, box_b = setup.box_mm(a), setup.box_mm(b)
    return [v for v in clip_convex(box_a, box_b)
            if boundary_distance(box_a, v) < 1e-6 and boundary_distance(box_b, v) < 1e-6]


def sample_grid(grid: ScreenGrid, field: np.ndarray, x_mm: float, y_mm: float) -> np.ndarray:
    """Value of a grid field at the grid pixel whose centre is nearest to (x_mm, y_mm)."""
    u, v = np.rint(apply_h(grid.mm_to_grid, np.array([x_mm, y_mm]))).astype(int)
    return field[v, u]
