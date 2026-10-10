"""Where things truly are in a camera frame, as polygons the pages draw on top of the picture.

When the projectors are aligned the blend hides the overlap completely: in bright content
nothing in the picture says where one projector stops and the other starts. So each sample frame
comes with its true geometry, mapped from screen millimetres to camera pixels through the
camera's view in that frame (which a knock changes):

  box A, box B   where each projector's raster lands now (its convex quadrilateral)
  overlap        box A ∩ box B
  content rect   where the calibration software puts the picture
  markers        the printed ArUco squares on the bezel, and whether the camera sees each
  hotspots       on a gain screen, where each projector's light looks brightest

Edge pieces are the boundary method's raw material (CLAUDE.md 4.2): each box outline cut into
pieces about EDGE_PIECE_MM long. A piece on the combined image's outer boundary always shows
against the unlit screen; a piece inside the other projector's box is an inner edge, visible only
through black level in dark frames or through the blend ramp. Each piece can only tell how far
its own edge moved across itself, along its outward normal.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from demo.renders import Family
from scripts.visualize import deepest_point, region_masks
from sim.calibration import CalibrationSetup
from sim.planar import apply_h, centroid, clip_convex, points_in_convex, rect_polygon


def _r(points: np.ndarray, digits: int = 1) -> list[list[float]]:
    return [[round(float(x), digits), round(float(y), digits)] for x, y in np.asarray(points).reshape(-1, 2)]


def rotate_cw(points: np.ndarray, height: int) -> np.ndarray:
    """Camera pixel (u, v) after turning the image 90° clockwise (cv2.ROTATE_90_CLOCKWISE)."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return np.stack([height - 1 - p[:, 1], p[:, 0]], axis=-1)


def layers(fam: Family, facts: dict[str, Any], labels: bool = True, rotate: bool = False) -> dict[str, Any]:
    """The true geometry of one frame in camera pixels (optionally turned 90° clockwise)."""
    w, h = fam.scene.camera.resolution
    cam = facts["camera_h"]

    def px(poly_mm: np.ndarray) -> list[list[float]]:
        p = apply_h(cam, np.asarray(poly_mm, dtype=np.float64))
        return _r(rotate_cw(p, h) if rotate else p)

    setup = fam.scene.setup
    box_a, box_b = facts["boxes_mm"]["a"], facts["boxes_mm"]["b"]
    overlap = clip_convex(box_a, box_b)
    out: dict[str, Any] = {
        "w": h if rotate else w, "h": w if rotate else h,
        "a": px(box_a), "b": px(box_b), "overlap": px(overlap) if len(overlap) >= 3 else [],
        "content": px(rect_polygon(*setup.content_rect_mm)), "markers": [], "hotspots": [], "labels": [],
    }
    markers = fam.scene.markers
    for i in range(0 if markers is None else len(markers.centres_mm)):
        out["markers"].append({"id": i, "poly": px(markers.square(i)), "visible": i in facts["markers_visible"]})
    if fam.scene.room.active:
        for name in setup.names:
            spot = fam.scene.room.hotspot_mm(name)
            if spot is not None:
                out["hotspots"].append({"name": name, "at": px(spot)[0]})
    if labels:
        pts, masks = region_masks(fam.scene, 0.0, facts["boxes_mm"])
        centre = tuple(np.vstack([box_a, box_b]).mean(axis=0))
        for text, region, layer in (("A only", "only_a", "a"), ("B only", "only_b", "b"), ("overlap", "overlap", "overlap")):
            found = deepest_point(pts, masks[region], centre)
            if found is not None:
                out["labels"].append({"text": text, "layer": layer, "at": px(found[0])[0]})
    return out


def edge_pieces(setup: CalibrationSetup, piece_mm: float = 80.0) -> list[dict[str, Any]]:
    """Each box's outline in pieces about `piece_mm` long: owner, ends, outward normal, outer or inner."""
    boxes = {n: setup.box_mm(n) for n in setup.names}
    content = rect_polygon(*setup.content_rect_mm)
    out = []
    for name, other in (("a", "b"), ("b", "a")):
        box = boxes[name]
        c = centroid(box)
        for k in range(len(box)):
            p, q = box[k], box[(k + 1) % len(box)]
            d = q - p
            length = float(np.hypot(*d))
            normal = np.array([d[1], -d[0]]) / length
            if np.dot(normal, (p + q) / 2 - c) < 0:
                normal = -normal  # point away from the box's own centre
            n = max(1, round(length / piece_mm))
            for j in range(n):
                p0, p1 = p + d * j / n, p + d * (j + 1) / n
                mid = (p0 + p1) / 2
                inner = bool(points_in_convex(boxes[other], mid, margin=0.5))
                out.append({"owner": name, "p0": _r(p0, 2)[0], "p1": _r(p1, 2)[0], "mid": _r(mid, 2)[0],
                            "normal": [round(float(normal[0]), 6), round(float(normal[1]), 6)],
                            "kind": "inner" if inner else "outer",
                            "in_content": bool(points_in_convex(content, mid, margin=-0.5))})
    return out


def geometry(setup: CalibrationSetup, screen_mm: tuple[float, float], piece_mm: float = 80.0) -> dict[str, Any]:
    """An arrangement's calibrated geometry in mm, for the boundary toy."""
    return {"screen": [float(screen_mm[0]), float(screen_mm[1])],
            "a": _r(setup.box_mm("a"), 2), "b": _r(setup.box_mm("b"), 2),
            "overlap": _r(setup.overlap(), 2), "content": [float(v) for v in setup.content_rect_mm],
            "pieces": edge_pieces(setup, piece_mm)}
