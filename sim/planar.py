"""Planar geometry for the simulator: homographies, convex polygons, and the one image warp.

(Also the scratch buffers that the warp and its callers reuse between frames.)

Everything here happens on a flat screen. Any mapping between two flat views of it -- projector
pixels to screen millimetres, screen millimetres to camera pixels -- is therefore a
*homography*: a 3x3 matrix that maps straight lines to straight lines. A projector's lit
footprint (its "box") is the image of its rectangular pixel raster, so it is a convex
quadrilateral, and the overlap of two boxes is a convex polygon.

This module is the simulator's own copy of that geometry. It deliberately shares no code with
``detector/``: ground truth must not inherit a bug from the thing it grades.

Conventions used throughout ``sim/``:
  * Pixel centres sit at integer coordinates (OpenCV's convention). A raster of width w and
    height h spans [-0.5, w-0.5] x [-0.5, h-0.5]; a box is the image of those outer corners.
  * Screen coordinates are millimetres, origin at the screen's top-left, x right, y down.
  * Polygons are (N, 2) float64 vertex arrays in order; either orientation is accepted.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

Segment = tuple[np.ndarray, np.ndarray]


def raster_corners(resolution: tuple[int, int]) -> np.ndarray:
    """Outer corners (TL, TR, BR, BL) of a w x h pixel raster in pixel coordinates."""
    w, h = resolution
    return np.array([[-0.5, -0.5], [w - 0.5, -0.5], [w - 0.5, h - 0.5], [-0.5, h - 0.5]])


def rect_polygon(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)


def translation(dx: float, dy: float) -> np.ndarray:
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])


def apply_h(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Map points (..., 2) through homography `h`."""
    pts = np.asarray(pts, dtype=np.float64)
    x, y = pts[..., 0], pts[..., 1]
    w = h[2, 0] * x + h[2, 1] * y + h[2, 2]
    return np.stack(
        [(h[0, 0] * x + h[0, 1] * y + h[0, 2]) / w, (h[1, 0] * x + h[1, 1] * y + h[1, 2]) / w], axis=-1
    )


def _normalizer(pts: np.ndarray) -> np.ndarray:
    """Similarity moving the points' centroid to 0 and their mean radius to sqrt(2) (Hartley)."""
    c = pts.mean(axis=0)
    s = np.sqrt(2.0) / np.mean(np.hypot(*(pts - c).T))
    return np.array([[s, 0.0, -s * c[0]], [0.0, s, -s * c[1]], [0.0, 0.0, 1.0]])


def homography_from_points(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """The exact homography taking 4 points `src` to 4 points `dst`, solved in float64.

    (OpenCV's getPerspectiveTransform works in float32, which costs ~5e-5 mm on a 4 m screen;
    ground truth should not carry that.)
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != (4, 2) or dst.shape != (4, 2):
        raise ValueError("homography_from_points needs exactly 4 point pairs")
    ts, td = _normalizer(src), _normalizer(dst)
    s, d = apply_h(ts, src), apply_h(td, dst)
    a = np.zeros((8, 8))
    b = np.zeros(8)
    for i, ((x, y), (u, v)) in enumerate(zip(s, d, strict=True)):
        a[2 * i] = [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y]
        a[2 * i + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y]
        b[2 * i], b[2 * i + 1] = u, v
    hn = np.append(np.linalg.solve(a, b), 1.0).reshape(3, 3)
    h = np.linalg.inv(td) @ hn @ ts
    return h / h[2, 2]


def box_mm(h_px_to_mm: np.ndarray, resolution: tuple[int, int]) -> np.ndarray:
    """A projector's box: its raster's outer corners mapped to screen millimetres."""
    return apply_h(h_px_to_mm, raster_corners(resolution))


def jacobian_det(h: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Local area magnification of homography `h` at (u, v): det(H) / w^3, w = h31 u + h32 v + h33."""
    w = h[2, 0] * u + h[2, 1] * v + h[2, 2]
    return np.linalg.det(h) / w**3


def local_scale(h: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Local linear scale of `h` at (u, v): the side of the image of a unit square, sqrt|det J|."""
    return np.sqrt(np.abs(jacobian_det(h, u, v)))


def is_vertical(poly: np.ndarray) -> bool:
    """True if the polygon's bounding box is at least as tall as it is wide (a side-by-side overlap).

    The marker layout and the zoomed camera both orient themselves by this one rule, so they
    always agree on where the overlap's ends are.
    """
    lo, hi = np.min(poly, axis=0), np.max(poly, axis=0)
    return bool(hi[1] - lo[1] >= hi[0] - lo[0])


def signed_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def centroid(poly: np.ndarray) -> np.ndarray:
    """Area centroid (centre of mass) of a simple polygon."""
    p = oriented(poly)
    x, y = p[:, 0], p[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    six_area = 3.0 * float(np.sum(cross))
    return np.array([float(np.sum((x + xn) * cross)) / six_area, float(np.sum((y + yn) * cross)) / six_area])


def oriented(poly: np.ndarray) -> np.ndarray:
    """The polygon with positive signed area, so its interior lies on the left of every edge."""
    poly = np.asarray(poly, dtype=np.float64)
    return poly if signed_area(poly) >= 0 else poly[::-1].copy()


def edges(poly: np.ndarray) -> list[Segment]:
    return [(poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))]


def is_convex(poly: np.ndarray) -> bool:
    p = oriented(poly)
    e = np.roll(p, -1, axis=0) - p
    turn = e[:, 0] * np.roll(e[:, 1], -1) - e[:, 1] * np.roll(e[:, 0], -1)
    return signed_area(p) > 0 and bool(np.all(turn >= -1e-9 * np.max(np.abs(p)) ** 2))


def edge_offsets(poly: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Signed distance of each point to each edge's line, positive inside: shape (..., n_edges)."""
    p = oriented(poly)
    e = np.roll(p, -1, axis=0) - p
    length = np.hypot(e[:, 0], e[:, 1])
    q = np.asarray(pts, dtype=np.float64)[..., None, :]
    return (e[:, 0] * (q[..., 1] - p[:, 1]) - e[:, 1] * (q[..., 0] - p[:, 0])) / length


def points_in_convex(poly: np.ndarray, pts: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """True where a point lies inside the convex polygon by at least `margin` (boundary counts at 0)."""
    return edge_offsets(poly, pts).min(axis=-1) >= margin


def clip_convex(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Part of polygon `subject` inside convex polygon `clip` (Sutherland-Hodgman). (0, 2) if empty."""
    out = list(oriented(subject))
    for a, b in edges(oriented(clip)):
        e = b - a
        side = [e[0] * (p[1] - a[1]) - e[1] * (p[0] - a[0]) for p in out]
        clipped = []
        for i in range(len(out)):
            prev, cur = out[i - 1], out[i]
            s_prev, s_cur = side[i - 1], side[i]
            if s_cur >= 0:
                if s_prev < 0:
                    clipped.append(prev + (cur - prev) * (s_prev / (s_prev - s_cur)))
                clipped.append(cur)
            elif s_prev >= 0:
                clipped.append(prev + (cur - prev) * (s_prev / (s_prev - s_cur)))
        out = clipped
        if not out:
            break
    return _dedup(np.array(out, dtype=np.float64).reshape(-1, 2))


def _dedup(poly: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Drop vertices that repeat their predecessor (clipping can emit a vertex twice)."""
    if len(poly) == 0:
        return poly
    keep = np.hypot(*(poly - np.roll(poly, 1, axis=0)).T) > tol
    poly = poly[keep] if keep.any() else poly[:1]
    return poly if len(poly) >= 3 else np.zeros((0, 2))


def segment_distance(pts: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from each point (..., 2) to the segment a-b."""
    pts = np.asarray(pts, dtype=np.float64)
    ab = b - a
    dx, dy = pts[..., 0] - a[0], pts[..., 1] - a[1]
    length2 = float(ab @ ab)
    t = np.zeros_like(dx) if length2 == 0 else np.clip((dx * ab[0] + dy * ab[1]) / length2, 0.0, 1.0)
    return np.hypot(dx - t * ab[0], dy - t * ab[1])


def distance_to_segments(pts: np.ndarray, segments: Sequence[Segment]) -> np.ndarray:
    """Distance from each point to the nearest of `segments`; +inf when there are none."""
    d = np.full(np.asarray(pts).shape[:-1], np.inf)
    for a, b in segments:
        d = np.minimum(d, segment_distance(pts, a, b))
    return d


def boundary_distance(poly: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return distance_to_segments(pts, edges(np.asarray(poly, dtype=np.float64)))


def scratch(workspace: dict | None, name: str, shape: tuple[int, ...]) -> np.ndarray | None:
    """A reusable float32 buffer of `shape` kept in `workspace` (None without a workspace).

    Rendering makes several grid- and sensor-sized temporaries per frame (hundreds of MB).
    Allocating them afresh each time costs page faults, which stall parallel workers; reusing
    buffers changes no value, because every operation writes all of its output.
    """
    if workspace is None:
        return None
    buf = workspace.get(name)
    if buf is None or buf.shape != tuple(shape):
        buf = workspace[name] = np.empty(shape, dtype=np.float32)
    return buf


def warp_linear(
    src: np.ndarray, m: np.ndarray, dsize: tuple[int, int], *, inverse: bool = False, border: float = 0.0,
    dst: np.ndarray | None = None,
) -> np.ndarray:
    """Bilinear perspective warp; `m` maps src -> dst pixel coordinates (dst -> src if `inverse`).

    Only float32 images with 1, 3 or 4 channels are accepted: for those OpenCV 5 interpolates at
    the exact sub-pixel position, while other types are rounded to 1/32 px -- a hidden error as
    large as 1/8 of the smallest shift the evaluation sweeps. `dst`, if given, receives the result.
    """
    if src.dtype != np.float32:
        raise TypeError(f"warp_linear needs float32 input, got {src.dtype}")
    channels = 1 if src.ndim == 2 else src.shape[2]
    if src.ndim not in (2, 3) or channels not in (1, 3, 4) or (src.ndim == 3 and channels == 1):
        raise ValueError(f"warp_linear needs an (h, w), (h, w, 3) or (h, w, 4) image, got {src.shape}")
    flags = cv2.INTER_LINEAR | (cv2.WARP_INVERSE_MAP if inverse else 0)
    return cv2.warpPerspective(
        src, np.asarray(m, dtype=np.float64), (int(dsize[0]), int(dsize[1])), dst=dst,
        flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=(border,) * 4,
    )
