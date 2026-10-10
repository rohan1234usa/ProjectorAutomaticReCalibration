"""Double contours, found the way CLAUDE.md 4.3 plans to find them: a teaching illustration.

When B has moved, every content feature in the overlap shows twice: O = a·S(x) + b·S(x - d). In
the Fourier domain that is S's spectrum times (a + b e^(-iωd)), a ripple of period 1/d, and the
log spectrum turns the product into a sum: log|O| = log|S| + log|a + b e^(-iωd)|. Transforming
the log spectrum back (the cepstrum) puts the ripple at one place, quefrency d: a peak at the
offset, whatever the content.

Text has its own repetitions -- line spacing, letter pitch -- which also make peaks. So each
core tile (where both blend weights are at least CORE_MIN_WEIGHT) is compared with control tiles
just outside the overlap, where A or B shines alone and the same kind of content shows no echo:

  diff = z(core) - (z(ctrl_a) + z(ctrl_b)) / 2,   z = robust z-score (median, 1.4826 MAD)

A peak counts if it stands MIN_PEAK_SNR above the noise and NULL_FACTOR above the null, the
largest |z(ctrl_a) - z(ctrl_b)|. A disk of ECHO_FLOOR_CAMERA_PX around the origin is blanked:
closer echoes merge with the origin peak (the floor). Tiles are cut on the camera grid.

Two simplifications make this an illustration, not ``detector/echo.py`` (Phase 6a). It knows from
the simulator which frames hold slides and pools the log spectra of all core tiles (and of all
control tiles) over the sixteen slides shown after B moved, instead of scoring each tile and
asking the tiles to agree on one smooth field. And its "unresolved" rule is its own: a peak that
passes both tests but is only the rising flank of something higher inside the floor is the rim of an
echo too close to the origin to measure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from demo import images as im
from demo.figures import save
from demo.manifest import ECHO_FRAMES, ECHO_SIZES, SHOTS, TWIN, across
from demo.renders import Family
from scripts.visualize import border_mm
from sim.planar import apply_h, boundary_distance, points_in_convex

WINDOW = 12  # cepstrum half-width searched and shown, camera px


def tile_size(fam: Family, tile_mm: float) -> tuple[int, float]:
    """Tile side in camera px (even) for TILE_MM at the overlap centre, and the camera px per mm there."""
    c = fam.scene.setup.overlap().mean(axis=0)
    h = fam.scene.camera.h_mm_to_px
    scale = float(np.linalg.norm(apply_h(h, c + [0.5, 0.0]) - apply_h(h, c - [0.5, 0.0])))
    return int(2 * round(tile_mm * scale / 2)), scale


def tiles(fam: Family, n: int, cfg: dict[str, Any]) -> dict[str, list[tuple[int, int]]]:
    """Top-left corners (camera px) of core tiles and of each one's control tiles beside the overlap."""
    setup, h = fam.scene.setup, fam.scene.camera.h_mm_to_px
    inv = np.linalg.inv(h)
    x0, y0, x1, y1 = setup.content_rect_mm
    m = border_mm(fam.scenarios[TWIN], SHOTS["echo_0"].frame) + 8.0
    picture = np.array([[x0 + m, y0 + m], [x1 - m, y0 + m], [x1 - m, y1 - m], [x0 + m, y1 - m]])
    corners = np.array([[0, 0], [n, 0], [n, n], [0, n], [n / 2, n / 2]], dtype=np.float64)

    def mm(u: float, v: float) -> np.ndarray:
        return apply_h(inv, corners + [u, v])

    def core(u: float, v: float) -> bool:
        pts = mm(u, v)
        w = setup.blend_at(pts)
        return bool(np.all(points_in_convex(picture, pts)) and np.all(np.minimum(w["a"], w["b"]) >= cfg["CORE_MIN_WEIGHT"]))

    ov = apply_h(h, setup.overlap())
    u_lo, u_hi, v_lo, v_hi = int(ov[:, 0].min()), int(ov[:, 0].max()), int(ov[:, 1].min()), int(ov[:, 1].max())
    v_start = next((v for v in range(v_lo, v_hi) if any(core(u, v) for u in range(u_lo, u_hi - n, 4))), None)
    if v_start is None:
        raise ValueError(f"no {n} px tile fits where both blend weights reach {cfg['CORE_MIN_WEIGHT']}")
    out: dict[str, list[tuple[int, int]]] = {"core": [], "ctrl_a": [], "ctrl_b": []}
    for v in range(v_start, v_hi - n, n):
        u_row = next((u for u in range(u_lo, u_hi - n) if core(u, v)), None)  # the keystone tilts the core's sides
        if u_row is None:
            continue
        for u in range(u_row, u_hi - n, n):
            if not core(u, v):
                break
            out["core"].append((u, v))
            centre = mm(u, v)[4]
            for name, other in (("ctrl_a", "b"), ("ctrl_b", "a")):
                g = setup.blend_at(centre[None] + [[1.0, 0.0], [0.0, 1.0]])[name[-1]] - setup.blend_at(centre[None])[name[-1]]
                g = g / np.hypot(*g)
                for s in np.arange(0.0, 600.0, 1.0):
                    p = apply_h(h, centre + s * g) - n / 2
                    pts = mm(*p)
                    outside = ~points_in_convex(setup.box_mm(other), pts)
                    if np.all(outside) and boundary_distance(setup.box_mm(other), pts).min() >= cfg["CTRL_MARGIN_MM"]:
                        if np.all(points_in_convex(picture, pts)):
                            out[name].append((int(round(p[0])), int(round(p[1]))))
                        break
    return out


def log_spectrum(tile: np.ndarray) -> np.ndarray:
    """Linear high-pass (the tile minus its blur), Hann window, log power."""
    t = tile.astype(np.float64)
    hp = t - cv2.GaussianBlur(t, (0, 0), 3.0, borderType=cv2.BORDER_REFLECT)
    w = np.outer(np.hanning(len(t)), np.hanning(len(t)))
    p = np.abs(np.fft.fft2(hp * w)) ** 2
    return np.log(p + 1e-6 * p.mean())


def zscore(c: np.ndarray) -> np.ndarray:
    med = np.median(c)
    return (c - med) / (1.4826 * np.median(np.abs(c - med)))


def _refine(win: np.ndarray, y: int, x: int) -> np.ndarray:
    """Sub-pixel peak position (parabola through each axis' three samples), relative to the window centre."""
    def vertex(a: float, b: float, c: float) -> float:
        d = a - 2 * b + c
        return 0.0 if d >= 0 else 0.5 * (a - c) / d

    dx = vertex(win[y, x - 1], win[y, x], win[y, x + 1]) if 0 < x < win.shape[1] - 1 else 0.0
    dy = vertex(win[y - 1, x], win[y, x], win[y + 1, x]) if 0 < y < win.shape[0] - 1 else 0.0
    return np.array([x - WINDOW + dx, y - WINDOW + dy])


def score(pools: dict[str, np.ndarray], floor_px: float, cfg: dict[str, Any]) -> dict[str, Any]:
    """Core-minus-control cepstrum near the origin, its best peak outside the floor, the null, the verdict.

    resolved: the peak passes both tests and is a peak of its own, outside the floor.
    unresolved: it passes, but a neighbour nearer the origin is higher: it is the rim of an echo
    inside the floor, too close to the origin to measure (present, size unknown).
    none: nothing passes; the echo, if any, is below the floor (the detector's None, upper bound = floor).
    """
    cep = {k: zscore(np.fft.fftshift(np.abs(np.fft.ifft2(v)))) for k, v in pools.items()}
    diff = cep["core"] - (cep["ctrl_a"] + cep["ctrl_b"]) / 2
    null_map = np.abs(cep["ctrl_a"] - cep["ctrl_b"])
    c = len(diff) // 2
    win = diff[c - WINDOW : c + WINDOW + 1, c - WINDOW : c + WINDOW + 1]
    yy, xx = np.mgrid[-WINDOW : WINDOW + 1, -WINDOW : WINDOW + 1]
    outside = np.hypot(xx, yy) > floor_px
    k = int(np.argmax(np.where(outside, win, -np.inf)))
    y, x = np.unravel_index(k, win.shape)
    peak = float(win[y, x])
    null = float(null_map[c - WINDOW : c + WINDOW + 1, c - WINDOW : c + WINDOW + 1][outside].max())
    at = _refine(win, y, x)
    if at[0] < 0 or (at[0] == 0 and at[1] < 0):
        at = -at  # the cepstrum is symmetric: the echo's sign is not known
    accepted = peak >= cfg["MIN_PEAK_SNR"] and peak >= cfg["NULL_FACTOR"] * null
    r0 = np.hypot(x - WINDOW, y - WINDOW)
    rim = any(win[y + dy, x + dx] > peak and np.hypot(x + dx - WINDOW, y + dy - WINDOW) < r0
              for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dx or dy) and 0 <= y + dy < len(win) and 0 <= x + dx < len(win))
    radius = float(np.hypot(*at))
    verdict = ("unresolved" if rim else "resolved") if accepted else "none"
    return {"peak": round(peak, 2), "at": [round(float(at[0]), 2), round(float(at[1]), 2)],
            "radius_px": round(radius, 2), "null": round(null, 2),
            "threshold": round(max(cfg["MIN_PEAK_SNR"], cfg["NULL_FACTOR"] * null), 2), "accepted": accepted,
            "verdict": verdict, "window": win}


def illustrate(out: Path, fam: Family, cfg: dict[str, Any]) -> dict[str, Any]:
    """Pool every core and control tile over the slides after the onset, for each shift size; score each."""
    n, scale = tile_size(fam, cfg["TILE_MM"])
    floor_px = max(cfg["ECHO_FLOOR_CAMERA_PX"], cfg["MIN_OFFSET_MM"] * scale)
    grid = tiles(fam, n, cfg)
    variants = {size: across(size) if size != "0" else TWIN for size in ECHO_SIZES}
    pools = {size: {k: np.zeros((n, n)) for k in grid} for size in ECHO_SIZES}
    for i in ECHO_FRAMES:  # frames outside, sizes inside: A's image and the picture are rendered once per frame
        for size, variant in variants.items():
            e = fam.electrons(variant, i)
            for k, corners in grid.items():
                for u, v in corners:
                    pools[size][k] += log_spectrum(e[v : v + n, u : u + n])
    results = []
    for size, variant in variants.items():
        res = score({k: pools[size][k] / (len(grid[k]) * len(ECHO_FRAMES)) for k in grid}, floor_px, cfg)
        facts = fam.facts(variant, ECHO_FRAMES[0])
        c = fam.scene.setup.overlap().mean(axis=0)
        shift = (facts["boxes_mm"]["b"] - fam.scene.setup.box_mm("b")).mean(axis=0)
        echo = apply_h(facts["camera_h"], c + shift) - apply_h(facts["camera_h"], c)
        results.append({"size_px": float(size), "offset_mm": round(facts["offset_mm"], 4),
                        "echo_px": [round(float(echo[0]), 2), round(float(echo[1]), 2)],
                        "estimate_mm": round(res["radius_px"] / scale, 3) if res["verdict"] == "resolved" else None,
                        "frames_per_tile": len(ECHO_FRAMES), **res})
    yy, xx = np.mgrid[-WINDOW : WINDOW + 1, -WINDOW : WINDOW + 1]
    blank = np.hypot(xx, yy) <= floor_px  # the method ignores the floor; so does the picture's colour scale
    vmax = max(float(r["window"][~blank].max()) for r in results)
    for r in results:
        shown = np.sqrt(np.clip(np.where(blank, 0.0, r.pop("window")), 0.0, None) / vmax)
        r["img"] = save(out, f"cepstrum_{r['size_px']:g}", im.ramp(shown, 1.0), "png")
    row = _row(fam, scale)
    return {"tile_px": n, "px_per_mm": round(scale, 4), "floor_px": round(floor_px, 3),
            "floor_mm": round(floor_px / scale, 3), "window": WINDOW, "vmax": round(vmax, 1),
            "tiles": {k: len(v) for k, v in grid.items()}, "frames": list(ECHO_FRAMES), "results": results,
            "row": row, "tile_corners": {k: [list(t) for t in v] for k, v in grid.items()}}


def _row(fam: Family, scale: float, length: int = 256) -> dict[str, Any]:
    """One camera row through the overlap (aligned), for the 1-D toy: the one with the most fine detail (text)."""
    i = SHOTS["echo_0"].frame
    e = fam.electrons(TWIN, i) / np.float32(fam.white)
    ov = apply_h(fam.scene.camera.h_mm_to_px, fam.scene.setup.overlap())
    u0 = int(ov[:, 0].mean()) - length // 2
    rows = range(int(ov[:, 1].min()) + 40, int(ov[:, 1].max()) - 40)
    v = max(rows, key=lambda r: float(np.var(np.diff(e[r, u0 : u0 + length]))))  # many sharp strokes, not one big edge
    return {"values": [round(float(x), 4) for x in e[v, u0 : u0 + length]], "v": v, "u0": u0,
            "px_per_mm": round(scale, 4)}
