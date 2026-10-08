"""The dataset checker's own plane geometry, written apart from sim/ so the check stays independent.

``scripts/check_dataset.py`` recomputes from setup.json what the simulator should have recorded.
If it borrowed the simulator's geometry (``sim/planar.py``, ``sim/truth.py``), a bug there would
pass its own check. So the few pieces it needs are written again here, the plain way:
homographies applied to points, the calibrated boxes, convex clipping for the overlap P, its
centroid, the projector pixel pitch there, the pivots, and a brute-force maximum of the
separation of A's and B's copies of the content over P.
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


class Geometry:
    def __init__(self, setup: dict[str, Any]) -> None:
        self.h = {n: np.array(p["h_cal_px_to_mm"]) for n, p in setup["projectors"].items()}
        self.res = {n: p["resolution"] for n, p in setup["projectors"].items()}
        self.boxes = {n: _box(self.h[n], self.res[n]) for n in self.h}
        x0, y0, x1, y1 = setup["content_rect_mm"]
        rect = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)
        self.overlap = _clip(_clip(self.boxes["a"], self.boxes["b"]), rect)
        self.pitch = max(_pitch(self.h[n], _centroid(self.overlap)) for n in self.h)

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
