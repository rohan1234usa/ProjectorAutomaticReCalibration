"""Test-side marker measurement: find the bezel markers in a simulated frame and locate their centres.

This is a measuring instrument for the simulator's tests, not detector code (the detector gets
its own in Phase 3). It answers the Phase 2 question "are the markers rendered so that a camera
can find them, and where they truly are?".

Two steps:
1. Identify. OpenCV's ArUco detector needs an 8-bit image. A linear 16-bit frame of a bright
   slide puts the room-lit paper at a few hundred electrons out of 20000, so a plain >> 8 or a
   min-max stretch crushes the markers to black. The frame is scaled so the paper sits near 200.
2. Locate. ArUco's own corners are biased and noisy in a single frame, so each of the black
   square's four outer edges is re-measured on the linear frame: many short profiles across
   the edge, each giving the sub-pixel point where the signal crosses halfway between ink and
   paper (unbiased for a symmetric blur), then a straight line through those points. Adjacent
   lines meet at the corners, and the centre is where the diagonals cross. A homography maps
   lines to lines, so that point is exactly the image of the marker's centre.
"""

from __future__ import annotations

import cv2
import numpy as np


def to_8bit(electrons: np.ndarray, paper_e: float) -> np.ndarray:
    """Scale linear electrons so white paper lands near 200 of 255."""
    return np.clip(np.rint(electrons * (200.0 / paper_e)), 0, 255).astype(np.uint8)


def detect(image8: np.ndarray, subpix: bool = False) -> dict[int, np.ndarray]:
    """ArUco DICT_4X4_50 corners (TL, TR, BR, BL in the marker's frame) by id.

    `subpix` turns on ArUco's own sub-pixel corner refinement, kept to show why
    :func:`refined_corners` exists (tests/test_fiducials.py compares the two).
    """
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX if subpix else cv2.aruco.CORNER_REFINE_NONE
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), params)
    corners, ids, _ = detector.detectMarkers(image8)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2).astype(np.float64) for i, c in zip(ids.ravel(), corners, strict=True)}


def _bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    h, w = img.shape
    x0, y0 = np.clip(x0, 0, w - 2), np.clip(y0, 0, h - 2)
    a, b = img[y0, x0], img[y0, x0 + 1]
    c, d = img[y0 + 1, x0], img[y0 + 1, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def _intersect(p: np.ndarray, d: np.ndarray, q: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Point where line p + s d meets line q + t e."""
    s = np.linalg.solve(np.column_stack([d, -e]), q - p)[0]
    return p + s * d


def refined_corners(electrons: np.ndarray, corners: np.ndarray, reach_px: float = 4.0) -> np.ndarray:
    """Refined corners of a marker's black square from straight-line fits to its four outer edges."""
    img = electrons.astype(np.float64)
    centre = corners.mean(axis=0)
    lines = []
    for k in range(4):
        p0, p1 = corners[k], corners[(k + 1) % 4]
        length = float(np.hypot(*(p1 - p0)))
        t = (p1 - p0) / length
        n = np.array([t[1], -t[0]])
        if n @ ((p0 + p1) / 2 - centre) < 0:
            n = -n  # outward: from ink (inside) to paper (outside)
        s = np.arange(0.15 * length, 0.85 * length, 0.5)
        tau = np.arange(-reach_px, reach_px + 1e-9, 0.25)
        pts = p0 + s[:, None] * t  # (m, 2)
        xs = pts[:, None, 0] + tau[None, :] * n[0]
        ys = pts[:, None, 1] + tau[None, :] * n[1]
        prof = _bilinear(img, xs, ys)  # (m, len(tau))
        inside, outside = prof[:, tau < -reach_px / 2].mean(axis=1), prof[:, tau > reach_px / 2].mean(axis=1)
        half = (inside + outside) / 2
        edge = []
        for row, level in zip(prof, half, strict=True):
            i = np.nonzero((row[:-1] < level) & (row[1:] >= level))[0]
            if len(i) != 1:
                edge.append(np.nan)  # noise made several crossings: skip this profile
                continue
            i = i[0]
            edge.append(tau[i] + (level - row[i]) * (tau[i + 1] - tau[i]) / (row[i + 1] - row[i]))
        edge = np.asarray(edge)
        ok = np.isfinite(edge)
        q = pts[ok] + edge[ok, None] * n
        mean = q.mean(axis=0)
        direction = np.linalg.svd(q - mean)[2][0]  # total least squares
        lines.append((mean, direction))
    return np.array([_intersect(*lines[k - 1], *lines[k]) for k in range(4)])


def centre_of(corners: np.ndarray) -> np.ndarray:
    """Where the square's diagonals cross: the image of its centre under any homography."""
    return _intersect(corners[0], corners[2] - corners[0], corners[1], corners[3] - corners[1])
