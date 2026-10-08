"""Arrangement presets: where each projector's box lands on the screen, and the content rect.

The two projectors may sit in any overlapping arrangement (CLAUDE.md decision 2): side by side,
one above the other, one rotated relative to the other, meeting at a corner, different sizes,
or one mostly on top of the other. A preset turns a few physical numbers (image widths, the
overlap, an angle) into each projector's *box*: the four screen-mm corners (TL, TR, BR, BL) of
its raster, which fix its calibrated homography. Pixels are square, so an image's height
follows from its width and the projector's resolution. Every pair is centred on the screen.

The content rect is the rectangle of screen the calibration software fills with the picture.
It must lie inside what the projectors cover, or part of the picture would land on nothing.
For axis-aligned boxes the largest such rectangle is written down exactly. For a rotated
projector it is found by a conservative search over 1 mm vertical strips: a convex box's top
edge is a convex function of x and its bottom edge a concave one, so within a strip the lowest
top and the highest bottom (what limits a rectangle there) sit at the strip's sides. The corner
arrangement uses the union's bounding box instead: the union is a staircase of two offset
boxes, no rectangle inside it crosses the overlap usefully, and content outside both boxes
simply goes unlit.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from sim.cfg import check_keys, choice, num, pair
from sim.planar import clip_convex, edges, is_convex, rect_polygon, signed_area

Resolutions = Mapping[str, tuple[int, int]]


@dataclass(frozen=True, eq=False)
class Arrangement:
    corners: dict[str, np.ndarray]  # TL, TR, BR, BL of each projector's box, screen mm
    content_rect_mm: tuple[float, float, float, float]  # (x0, y0, x1, y1)


def _box(x: float, y: float, w: float, h: float) -> np.ndarray:
    return rect_polygon(x, y, x + w, y + h)


def _height(width_mm: float, resolution: tuple[int, int]) -> float:
    return width_mm * resolution[1] / resolution[0]


def side_by_side(
    screen_mm: tuple[float, float], res: Resolutions, width_a_mm: float, width_b_mm: float, overlap_mm: float,
    vertical_offset_mm: float = 0.0, align: str = "top",
) -> Arrangement:
    """A on the left, B on the right, overlapping by `overlap_mm`.

    B's top sits `vertical_offset_mm` below A's top (``align: top``) or below the height that
    centres it on A (``align: centre``). Content: the full combined width over the common height.
    """
    if not 0 < overlap_mm < min(width_a_mm, width_b_mm):
        raise ValueError("arrangement: overlap must be positive and smaller than both image widths")
    ha, hb = _height(width_a_mm, res["a"]), _height(width_b_mm, res["b"])
    total_w = width_a_mm + width_b_mm - overlap_mm
    xa = (screen_mm[0] - total_w) / 2
    yb_rel = vertical_offset_mm + (0.0 if align == "top" else (ha - hb) / 2)
    top, bottom = min(0.0, yb_rel), max(ha, yb_rel + hb)
    ya = (screen_mm[1] - (bottom - top)) / 2 - top
    yb = ya + yb_rel
    content = (xa, max(ya, yb), xa + total_w, min(ya + ha, yb + hb))
    return Arrangement({"a": _box(xa, ya, width_a_mm, ha), "b": _box(xa + width_a_mm - overlap_mm, yb, width_b_mm, hb)},
                       content)


def stacked(
    screen_mm: tuple[float, float], res: Resolutions, width_a_mm: float, width_b_mm: float, overlap_mm: float,
    horizontal_offset_mm: float = 0.0,
) -> Arrangement:
    """A on top, B below, overlapping by `overlap_mm`; B centred under A, then shifted right."""
    ha, hb = _height(width_a_mm, res["a"]), _height(width_b_mm, res["b"])
    if not 0 < overlap_mm < min(ha, hb):
        raise ValueError("arrangement: overlap must be positive and smaller than both image heights")
    xb_rel = (width_a_mm - width_b_mm) / 2 + horizontal_offset_mm
    left, right = min(0.0, xb_rel), max(width_a_mm, xb_rel + width_b_mm)
    xa = (screen_mm[0] - (right - left)) / 2 - left
    xb = xa + xb_rel
    total_h = ha + hb - overlap_mm
    ya = (screen_mm[1] - total_h) / 2
    content = (max(xa, xb), ya, min(xa + width_a_mm, xb + width_b_mm), ya + total_h)
    return Arrangement({"a": _box(xa, ya, width_a_mm, ha), "b": _box(xb, ya + ha - overlap_mm, width_b_mm, hb)},
                       content)


def rotated(
    screen_mm: tuple[float, float], res: Resolutions, width_mm: float, overlap_mm: float, angle_deg: float,
    vertical_offset_mm: float = 0.0,
) -> Arrangement:
    """Side by side, then B turned by `angle_deg` about its centre (positive turns +x toward +y)."""
    base = side_by_side(screen_mm, res, width_mm, width_mm, overlap_mm, vertical_offset_mm)
    b = base.corners["b"]
    c = b.mean(axis=0)
    t = math.radians(angle_deg)
    rot = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    corners = {"a": base.corners["a"], "b": (b - c) @ rot.T + c}
    union = np.vstack(list(corners.values()))
    shift = np.array(screen_mm) / 2 - (union.min(axis=0) + union.max(axis=0)) / 2  # re-centre the pair
    corners = {n: q + shift for n, q in corners.items()}
    return Arrangement(corners, largest_rect_in_union(list(corners.values())))


def corner(
    screen_mm: tuple[float, float], res: Resolutions, width_mm: float, overlap_x_mm: float, overlap_y_mm: float
) -> Arrangement:
    """A top-left, B bottom-right, meeting in an `overlap_x_mm` x `overlap_y_mm` corner overlap."""
    ha, hb = _height(width_mm, res["a"]), _height(width_mm, res["b"])
    if not (0 < overlap_x_mm < width_mm and 0 < overlap_y_mm < min(ha, hb)):
        raise ValueError("arrangement: corner overlap must be positive and smaller than both images")
    total_w = 2 * width_mm - overlap_x_mm
    total_h = ha + hb - overlap_y_mm
    xa, ya = (screen_mm[0] - total_w) / 2, (screen_mm[1] - total_h) / 2
    b = _box(xa + width_mm - overlap_x_mm, ya + ha - overlap_y_mm, width_mm, hb)
    return Arrangement({"a": _box(xa, ya, width_mm, ha), "b": b}, (xa, ya, xa + total_w, ya + total_h))


def _column_interval(poly: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Top and bottom of convex `poly` along vertical lines at `x` (+inf/-inf where it is absent)."""
    lo, hi = np.full(x.shape, np.inf), np.full(x.shape, -np.inf)
    for p, q in edges(poly):
        if abs(q[0] - p[0]) < 1e-12:
            on = np.abs(x - p[0]) < 1e-9
            lo = np.where(on, np.minimum(lo, min(p[1], q[1])), lo)
            hi = np.where(on, np.maximum(hi, max(p[1], q[1])), hi)
            continue
        on = (x >= min(p[0], q[0]) - 1e-9) & (x <= max(p[0], q[0]) + 1e-9)
        y = p[1] + (x - p[0]) * (q[1] - p[1]) / (q[0] - p[0])
        lo, hi = np.where(on, np.minimum(lo, y), lo), np.where(on, np.maximum(hi, y), hi)
    return lo, hi


def largest_rect_in_union(polys: list[np.ndarray], step_mm: float = 1.0) -> tuple[float, float, float, float]:
    """A (near) largest axis-aligned rectangle inside the union of convex polygons; never outside it."""
    pts = np.vstack(polys)
    n = math.ceil((pts[:, 0].max() - pts[:, 0].min()) / step_mm)
    xs = pts[:, 0].min() + step_mm * np.arange(n + 1)
    lo, hi = np.full(n, np.inf), np.full(n, -np.inf)
    for poly in polys:  # per strip: the part of each polygon that spans it, merged when they touch
        top, bottom = _column_interval(poly, xs)
        p_lo, p_hi = np.maximum(top[:-1], top[1:]), np.minimum(bottom[:-1], bottom[1:])
        ok = np.isfinite(p_lo) & np.isfinite(p_hi) & (p_hi > p_lo)
        touch = ok & (p_lo <= hi) & (lo <= p_hi)
        longer = ok & ~touch & (p_hi - p_lo > hi - lo)
        lo = np.where(touch, np.minimum(lo, p_lo), np.where(longer, p_lo, lo))
        hi = np.where(touch, np.maximum(hi, p_hi), np.where(longer, p_hi, hi))
    best, rect = 0.0, None
    for i in range(n):
        top, bottom = np.maximum.accumulate(lo[i:]), np.minimum.accumulate(hi[i:])
        area = np.where(bottom > top, (xs[i + 1:] - xs[i]) * (bottom - top), 0.0)
        j = int(np.argmax(area))
        if area[j] > best:
            best, rect = float(area[j]), (float(xs[i]), float(top[j]), float(xs[i + j + 1]), float(bottom[j]))
    if rect is None:
        raise ValueError("arrangement: the boxes leave no room for a content rect")
    return rect


Preset = tuple[Callable[..., Arrangement], dict[str, Any]]
PRESETS: dict[str, Preset] = {  # defaults fit the demo screen: 4000 x 1500 mm, 1920 x 1080 projectors
    "side_by_side": (side_by_side, {"width_mm": 2000.0, "overlap_mm": 410.0, "vertical_offset_mm": 0.3, "align": "top"}),
    "stacked": (stacked, {"width_mm": 1400.0, "overlap_mm": 200.0, "horizontal_offset_mm": 0.3}),
    "rotated": (rotated, {"width_mm": 2000.0, "overlap_mm": 410.0, "angle_deg": 1.0, "vertical_offset_mm": 0.0}),
    "corner": (corner, {"width_mm": 1600.0, "overlap_x_mm": 400.0, "overlap_y_mm": 400.0}),
    "different_sizes": (side_by_side, {"width_a_mm": 2000.0, "width_b_mm": 1600.0, "overlap_mm": 410.0,
                                       "vertical_offset_mm": 0.0, "align": "centre"}),
    "large_overlap": (side_by_side, {"width_mm": 2000.0, "overlap_mm": 1500.0, "vertical_offset_mm": 0.3,
                                     "align": "top"}),
}
_PER_PROJECTOR_WIDTH = {"side_by_side", "stacked", "different_sizes", "large_overlap"}


def from_config(cfg: Mapping[str, Any], screen_mm: tuple[float, float], res: Resolutions) -> Arrangement:
    """Expand a scenario's ``arrangement`` block into boxes and a content rect, and validate them."""
    cfg = dict(cfg)
    preset = cfg.pop("preset", None)
    if preset == "explicit":
        check_keys(cfg, {"corners_mm", "content_rect_mm"}, "arrangement")
        if "corners_mm" not in cfg or "content_rect_mm" not in cfg:
            raise ValueError("arrangement explicit: needs corners_mm and content_rect_mm")
        corners = {n: np.array([pair(p, f"arrangement.corners_mm.{n}") for p in c]) for n, c in cfg["corners_mm"].items()}
        if set(corners) != set(res) or any(c.shape != (4, 2) for c in corners.values()):
            raise ValueError("arrangement.corners_mm: 4 corners (TL, TR, BR, BL) for each projector")
        rect = cfg["content_rect_mm"]
        if len(rect) != 4:
            raise ValueError("arrangement.content_rect_mm: expected [x0, y0, x1, y1]")
        x0, y0, x1, y1 = (num(v, "arrangement.content_rect_mm") for v in rect)
        arrangement = Arrangement(corners, (x0, y0, x1, y1))
    elif preset in PRESETS:
        make, defaults = PRESETS[preset]
        allowed = set(defaults) | ({"width_mm", "width_a_mm", "width_b_mm"} if preset in _PER_PROJECTOR_WIDTH else set())
        check_keys(cfg, allowed, f"arrangement ({preset})")
        params = {**defaults, **cfg}
        if preset in _PER_PROJECTOR_WIDTH:  # precedence: width_a_mm > width_mm, the scenario's > the preset's
            for n in ("a", "b"):
                key = f"width_{n}_mm"
                params[key] = cfg.get(key, cfg.get("width_mm", defaults.get(key, defaults.get("width_mm"))))
            params.pop("width_mm", None)
        if "align" in params:
            params["align"] = choice(params["align"], ("top", "centre"), "arrangement.align")
        params = {k: v if k == "align" else num(v, f"arrangement.{k}") for k, v in params.items()}
        arrangement = make(screen_mm, res, **params)
    else:
        raise ValueError(f"arrangement.preset must be one of {sorted([*PRESETS, 'explicit'])}, got {preset!r}")
    validate(arrangement, screen_mm)
    return arrangement


def validate(arrangement: Arrangement, screen_mm: tuple[float, float]) -> None:
    """Boxes convex and on the screen; a content rect on the screen that crosses the overlap."""
    w, h = screen_mm
    tol = 1e-6
    for name, c in arrangement.corners.items():
        if not is_convex(c):
            raise ValueError(f"arrangement: box {name} is not a convex quadrilateral")
        if c[:, 0].min() < -tol or c[:, 1].min() < -tol or c[:, 0].max() > w + tol or c[:, 1].max() > h + tol:
            raise ValueError(f"arrangement: box {name} does not fit on the {w:g} x {h:g} mm screen")
    x0, y0, x1, y1 = arrangement.content_rect_mm
    if not (x1 > x0 and y1 > y0) or x0 < -tol or y0 < -tol or x1 > w + tol or y1 > h + tol:
        raise ValueError(f"arrangement: content rect {arrangement.content_rect_mm} must lie on the screen")
    a, b = arrangement.corners["a"], arrangement.corners["b"]
    overlap = clip_convex(clip_convex(a, b), rect_polygon(x0, y0, x1, y1))
    if len(overlap) == 0 or abs(signed_area(overlap)) <= 0:
        raise ValueError("arrangement: the content does not cross the overlap of the two boxes")
