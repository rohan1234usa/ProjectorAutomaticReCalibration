"""The dataset checker's own plane geometry, written apart from sim/ so the check stays independent.

``scripts/check_dataset.py`` recomputes from setup.json what the simulator should have recorded.
If it borrowed the simulator's geometry (``sim/planar.py``, ``sim/truth.py``), a bug there would
pass its own check. So the few pieces it needs are written again here, the plain way:
homographies applied to points, the calibrated boxes, convex clipping for the overlap P, its
centroid, the projector pixel pitch there, the pivots, the direction a shift "across" means (the
length-weighted normal of the moving projector's inner edges: the overlap's edges on its own
outline only, off the content rect's, where calibration fades it out; pointing away from its
partner), and a brute-force maximum of the separation of A's and B's copies of the content over P.

It also rebuilds what each recorded homography should be: a perturbation is a screen-mm
homography M applied after calibration, h_actual = M_n ... M_1 h_cal in the order listed, with
M a translation, or a rotation, uniform scale or keystone about a pivot (the conventions of the
scenario format: a positive angle turns +x toward +y, a keystone k along unit axis v has the
bottom row [k v, 1]); the relative homography h_rel = D_B D_A^-1 with D_p = h_actual,p h_cal,p^-1;
a knocked camera's homography, the unmoved one turned and shifted in the image; and which
markers' paper lies wholly inside the frame.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _apply(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.atleast_2d(pts)
    w = h[2, 0] * pts[:, 0] + h[2, 1] * pts[:, 1] + h[2, 2]
    return np.stack([(h[0, 0] * pts[:, 0] + h[0, 1] * pts[:, 1] + h[0, 2]) / w,
                     (h[1, 0] * pts[:, 0] + h[1, 1] * pts[:, 1] + h[1, 2]) / w], axis=-1)


def _box(h: np.ndarray, resolution: list[int]) -> np.ndarray:
    w, hh = resolution
    return _apply(h, np.array([[-0.5, -0.5], [w - 0.5, -0.5], [w - 0.5, hh - 0.5], [-0.5, hh - 0.5]]))


def _ccw(poly: np.ndarray) -> np.ndarray:
    area = np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
    return poly if area > 0 else poly[::-1]


def _clip(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman: the part of convex `subject` inside convex `clip`."""
    out = list(_ccw(subject))
    clip = _ccw(clip)
    for a, b in zip(clip, np.roll(clip, -1, axis=0), strict=True):
        inside = [(b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0 for p in out]
        new = []
        for i, p in enumerate(out):
            q, q_in = out[i - 1], inside[i - 1]
            if inside[i] != q_in:
                d = p - q
                e = b - a
                t = ((a[0] - q[0]) * e[1] - (a[1] - q[1]) * e[0]) / (d[0] * e[1] - d[1] * e[0])
                new.append(q + t * d)
            if inside[i]:
                new.append(p)
        out = new
    return np.array(out)


def _centroid(poly: np.ndarray) -> np.ndarray:
    x, y = poly[:, 0], poly[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    return np.array([np.sum((x + np.roll(x, -1)) * cross), np.sum((y + np.roll(y, -1)) * cross)]) / (3 * np.sum(cross))


def _pitch(h: np.ndarray, at_mm: np.ndarray) -> float:
    u, v = _apply(np.linalg.inv(h), at_mm)[0]
    w = h[2, 0] * u + h[2, 1] * v + h[2, 2]
    return math.sqrt(abs(np.linalg.det(h) / w**3))


def _on_outline(poly: np.ndarray, pt: np.ndarray, tol_mm: float = 1e-6) -> bool:
    """True if pt lies on the polygon's outline (within float rounding)."""
    for a, b in zip(poly, np.roll(poly, -1, axis=0), strict=True):
        ab = b - a
        t = min(1.0, max(0.0, float((pt - a) @ ab) / float(ab @ ab)))
        if math.hypot(*(pt - a - t * ab)) < tol_mm:
            return True
    return False


class Geometry:
    def __init__(self, setup: dict[str, Any]) -> None:
        self.h = {n: np.array(p["h_cal_px_to_mm"]) for n, p in setup["projectors"].items()}
        self.res = {n: p["resolution"] for n, p in setup["projectors"].items()}
        self.boxes = {n: _box(self.h[n], self.res[n]) for n in self.h}
        x0, y0, x1, y1 = setup["content_rect_mm"]
        self.rect = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)
        self.overlap = _clip(_clip(self.boxes["a"], self.boxes["b"]), self.rect)
        self.pitch = max(_pitch(self.h[n], _centroid(self.overlap)) for n in self.h)

    def across(self, name: str) -> np.ndarray | None:
        """Unit vector across `name`'s inner edges, away from its partner; None if it has none."""
        partner = "b" if name == "a" else "a"
        total, length = np.zeros(2), 0.0
        for p, q in zip(self.overlap, np.roll(self.overlap, -1, axis=0), strict=True):
            mid = (p + q) / 2
            if (_on_outline(self.boxes[name], mid) and not _on_outline(self.boxes[partner], mid)
                    and not _on_outline(self.rect, mid)):
                normal = np.array([q[1] - p[1], p[0] - q[0]])  # as long as the edge
                if normal @ (mid - self.boxes[name].mean(axis=0)) < 0:
                    normal = -normal  # outward from the box, into the partner
                total += normal
                length += math.hypot(*normal)
        norm = math.hypot(*total)
        return None if norm < 1e-6 * length or length == 0 else -total / norm

    def pivot(self, name: str, spec: Any) -> np.ndarray:
        if spec in (None, "centre"):
            w, h = self.res[name]
            return _apply(self.h[name], np.array([(w - 1) / 2, (h - 1) / 2]))[0]
        if spec == "overlap_centre":
            return _centroid(self.overlap)
        if spec == "far_corner":
            d = np.hypot(*(self.boxes[name] - _centroid(self.overlap)).T)
            return self.boxes[name][int(np.nonzero(d >= d.max() - 1e-9)[0][0])]
        return np.array(spec, dtype=float)

    def reach(self, pivot: np.ndarray) -> float:
        return float(np.hypot(*(self.overlap - pivot).T).max())


def brute_force_mm(line: dict[str, Any], geo: Geometry) -> float:
    d = {n: np.array(line["truth"]["h_actual"][n]) @ np.linalg.inv(geo.h[n]) for n in ("a", "b")}
    poly = geo.overlap
    lo, hi = poly.min(axis=0), poly.max(axis=0)
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 401), np.linspace(lo[1], hi[1], 401))
    pts = np.stack([gx.ravel(), gy.ravel()], axis=-1)
    inside = np.ones(len(pts), bool)
    for a, b in zip(_ccw(poly), np.roll(_ccw(poly), -1, axis=0), strict=True):
        inside &= (b[0] - a[0]) * (pts[:, 1] - a[1]) - (b[1] - a[1]) * (pts[:, 0] - a[0]) >= -1e-9
    edge_pts = [p + np.linspace(0, 1, 2001)[:, None] * (q - p) for p, q in zip(poly, np.roll(poly, -1, axis=0), strict=True)]
    pts = np.vstack([pts[inside], poly, *edge_pts])
    return float(np.hypot(*(_apply(d["b"], pts) - _apply(d["a"], pts)).T).max())


def translation(dx: float, dy: float) -> np.ndarray:
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])


def transform(kind: str, value: float, vector: np.ndarray, pivot: np.ndarray) -> np.ndarray:
    """One perturbation as a screen-mm homography: shift by value x vector, or the rest about pivot."""
    if kind == "shift":
        return translation(*(value * np.asarray(vector, dtype=float)))
    if kind == "rotation":
        c, s = math.cos(value), math.sin(value)
        local = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    elif kind == "scale":
        local = np.diag([1.0 + value, 1.0 + value, 1.0])
    else:  # keystone
        local = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [value * vector[0], value * vector[1], 1.0]])
    return translation(*pivot) @ local @ translation(-pivot[0], -pivot[1])


def compose(h_cal: np.ndarray, moves: list[np.ndarray]) -> np.ndarray:
    """h_cal after the moves, the first listed applied first."""
    out = h_cal
    for move in moves:
        out = move @ out
    return out


def relative_homography(h_actual: dict[str, np.ndarray], h_cal: dict[str, np.ndarray]) -> np.ndarray:
    """B's displacement relative to A's, D_B D_A^-1, normalised so that h[2, 2] = 1."""
    d = {n: np.eye(3) if np.array_equal(h_actual[n], h_cal[n]) else h_actual[n] @ np.linalg.inv(h_cal[n])
         for n in ("a", "b")}
    h = d["b"] @ np.linalg.inv(d["a"])
    return h / h[2, 2]


def knocked(h_mm_to_px: np.ndarray, resolution: list[int], shift_px: list[float], rotation_deg: float,
            m: float) -> np.ndarray:
    """The camera homography after a knock of strength m: the image turned about its centre, then shifted."""
    w, h = resolution
    cx, cy = (w - 1) / 2, (h - 1) / 2
    t = math.radians(m * rotation_deg)
    turn = np.array([[math.cos(t), -math.sin(t), 0.0], [math.sin(t), math.cos(t), 0.0], [0.0, 0.0, 1.0]])
    image = translation(m * shift_px[0], m * shift_px[1]) @ translation(cx, cy) @ turn @ translation(-cx, -cy)
    return image @ h_mm_to_px


def markers_in_view(setup: dict[str, Any], h_mm_to_px: np.ndarray, margin_px: float = 2.0) -> list[int]:
    """Ids of the markers whose whole printed paper lies inside the frame, margin_px from its edges."""
    markers = setup.get("markers")
    if not markers:
        return []
    w, h = setup["camera"]["resolution"]
    size = markers["size_mm"]
    half = (size + 2 * markers["quiet_zone_cells"] * size / 6) / 2  # 6 cells across the black square
    out = []
    for item in markers["items"]:
        x, y = item["centre_mm"]
        paper = _apply(h_mm_to_px, np.array([[x - half, y - half], [x + half, y - half],
                                             [x + half, y + half], [x - half, y + half]]))
        lo, hi = paper.min(axis=0), paper.max(axis=0)
        if lo[0] >= margin_px - 0.5 and lo[1] >= margin_px - 0.5 and hi[0] <= w - 0.5 - margin_px \
                and hi[1] <= h - 0.5 - margin_px:
            out.append(item["id"])
    return out
