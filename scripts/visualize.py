"""Quick-look image of one simulated camera frame, for humans.

Renders frame 0 of a scenario and writes ``view.png``: the camera frame shown with a
logarithmic display curve,

    y = ln(1 + x/b) / ln(1 + 1/b),

where x is the radiance relative to one projector's full white and b is the projector black
level. A plain sRGB display would put the black level (1/1500 of white) at about 3/255, which
is invisible. The log curve is roughly how a dark-adapted eye responds, and shows the bright
content and the faint black-level raster at the same time.

Below it, a second panel shows the same frame dimmed, with the scenario's true geometry drawn
on top: each projector's box (its lit raster), the overlap polygon where both shine, and the
content rect. When aligned, the blend makes the overlap invisible in bright content, so without
this panel it is easy to mistake a content feature for a projector boundary.

It also prints one JSON line: render time, and the levels measured in small patches of the
frame -- the unlit screen, each projector's black, and the overlap's doubled black (the black
patches need content with a black border). Each patch sits at the point deepest inside its
true region, so it works for any arrangement; a region that does not exist gives ``null``.
Using the true geometry is fine here: this is harness code, not the detector.

Usage: python -m scripts.visualize scenarios/aligned_side_by_side.yaml --out out/phase1
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from sim.content import border_px
from sim.planar import apply_h, clip_convex, points_in_convex, rect_polygon
from sim.render import QUALITY
from sim.scenario import Scene, load_scene

Regions = tuple[np.ndarray, dict[str, np.ndarray]]  # sample points (rows, cols, 2) and region masks

# Overlay colours (RGB).
_COLOR_A = (255, 150, 30)
_COLOR_B = (60, 170, 255)
_COLOR_OVERLAP = (60, 220, 90)
_COLOR_CONTENT = (235, 235, 235)
_REGION_STEP_MM = 2.0  # sampling step of the region masks used to place labels and patches


def log_display(x: np.ndarray, black: float) -> np.ndarray:
    """Map relative radiance to 8-bit display values with the log curve above."""
    y = np.log1p(np.maximum(x, 0.0) / black) / math.log1p(1.0 / black)
    return np.clip(np.rint(255.0 * y), 0, 255).astype(np.uint8)


def white_electrons(scene: Scene) -> float:
    """Electrons recorded (image centre) for one projector's full white on the screen."""
    brightness = scene.projectors[scene.setup.names[0]].brightness
    return scene.screen.reflectance * brightness * scene.camera.exposure * scene.camera.full_well_e


def border_mm(scene: Scene) -> float:
    """Width of the content's black border on the screen (0 when the content has none)."""
    _, y0, _, y1 = scene.setup.content_rect_mm
    size = scene.setup.content_size()
    return border_px(scene.content, size) * (y1 - y0) / size[1]


def region_masks(scene: Scene) -> Regions:
    """Screen sample points (rows, cols, 2) and boolean masks of the regions that matter.

    only_a / only_b / overlap: lit by one or both projectors. unlit: on the screen, outside both
    rasters. black_*: inside the content rect but in its black border, split by who lights it.
    """
    w, h = scene.screen.size_mm
    xs = np.arange(_REGION_STEP_MM / 2, w, _REGION_STEP_MM)
    ys = np.arange(_REGION_STEP_MM / 2, h, _REGION_STEP_MM)
    pts = np.stack(np.meshgrid(xs, ys), axis=-1)
    a, b = scene.setup.names
    in_a = points_in_convex(scene.setup.box_mm(a), pts)
    in_b = points_in_convex(scene.setup.box_mm(b), pts)
    x0, y0, x1, y1 = scene.setup.content_rect_mm
    m = border_mm(scene)
    x, y = pts[..., 0], pts[..., 1]
    in_content = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    in_picture = (x >= x0 + m) & (x <= x1 - m) & (y >= y0 + m) & (y <= y1 - m)
    border = in_content & ~in_picture
    return pts, {
        "only_a": in_a & ~in_b,
        "only_b": in_b & ~in_a,
        "overlap": in_a & in_b,
        "unlit": ~in_a & ~in_b,
        "black_a": in_a & ~in_b & border,
        "black_b": in_b & ~in_a & border,
        "black_overlap": in_a & in_b & border,
    }


def deepest_point(pts: np.ndarray, mask: np.ndarray, toward: tuple[float, float]) -> tuple[np.ndarray, float] | None:
    """The point of `mask` farthest from its edges, and that distance in mm; None if empty.

    Exact ties (e.g. along a band of constant width) go to the point nearest `toward`.
    """
    if not mask.any():
        return None
    padded = np.pad(mask, 1).astype(np.uint8)  # the screen's own edge also bounds every region
    depth = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1] * _REGION_STEP_MM
    score = np.where(mask, depth - 1e-4 * np.hypot(pts[..., 0] - toward[0], pts[..., 1] - toward[1]), -np.inf)
    r, c = np.unravel_index(int(np.argmax(score)), mask.shape)
    return pts[r, c], float(depth[r, c])


def measure_levels(
    scene: Scene, electrons: np.ndarray, regions: Regions | None = None
) -> dict[str, dict[str, float] | None]:
    """Mean and per-pixel spread (electrons, all channels) in a small square at each patch."""
    pts, masks = regions or region_masks(scene)
    centre = (scene.screen.size_mm[0] / 2, scene.screen.size_mm[1] / 2)
    px_per_mm = scene.camera.px_per_mm_at_centre()
    frame_h, frame_w = electrons.shape[:2]
    out: dict[str, dict[str, float] | None] = {}
    for name in ("unlit", "black_a", "black_b", "black_overlap"):
        found = deepest_point(pts, masks[name], centre)
        if found is None or found[1] * px_per_mm < 3.0:
            out[name] = None  # region missing or too thin to hold a patch
            continue
        point, depth_mm = found
        half = max(1, int(0.5 * depth_mm * px_per_mm))  # the square stays inside the inscribed circle
        u, v = np.rint(apply_h(scene.camera.h_mm_to_px, point)).astype(int)
        if not (half <= u < frame_w - half and half <= v < frame_h - half):
            out[name] = None  # the camera does not see this patch whole
            continue
        patch = electrons[v - half : v + half + 1, u - half : u + half + 1]
        out[name] = {"mean_e": round(float(patch.mean()), 2), "std_e": round(float(patch.std()), 2),
                     "at_mm": [round(float(point[0]), 1), round(float(point[1]), 1)]}
    return out


def _fixed_point(poly_px: np.ndarray) -> np.ndarray:
    """Pixel coordinates with 4 fractional bits, for OpenCV's sub-pixel drawing (shift=4)."""
    return np.rint(np.asarray(poly_px) * 16).astype(np.int32)


def _dashed(img: np.ndarray, poly_px: np.ndarray, color: tuple[int, int, int], thickness: int, dash: float) -> None:
    """Closed dashed outline, so a second outline lying on top of another stays distinguishable."""
    for p, q in zip(poly_px, np.roll(poly_px, -1, axis=0)):
        length = float(np.hypot(*(q - p)))
        for s in np.arange(0.0, length, 2 * dash):
            a = p + (q - p) * (s / length)
            b = p + (q - p) * (min(s + dash, length) / length)
            cv2.line(img, tuple(_fixed_point(a)), tuple(_fixed_point(b)), color, thickness, cv2.LINE_AA, shift=4)


def overlay_panel(scene: Scene, view: np.ndarray, regions: Regions | None = None) -> np.ndarray:
    """The frame dimmed to gray, with box A, box B, their overlap and the content rect drawn on it."""
    img = np.repeat((view.mean(axis=2, keepdims=True) * 0.55).astype(np.uint8), 3, axis=2)
    h_cam = scene.camera.h_mm_to_px
    a, b = scene.setup.names
    box_a, box_b = scene.setup.box_mm(a), scene.setup.box_mm(b)
    overlap_px = apply_h(h_cam, clip_convex(box_a, box_b))
    t = max(2, img.shape[1] // 900)

    if len(overlap_px):
        fill = img.copy()
        cv2.fillPoly(fill, [_fixed_point(overlap_px)], _COLOR_OVERLAP, cv2.LINE_AA, shift=4)
        img = cv2.addWeighted(fill, 0.35, img, 0.65, 0.0)
        cv2.polylines(img, [_fixed_point(overlap_px)], True, _COLOR_OVERLAP, max(1, t // 2), cv2.LINE_AA, shift=4)
    cv2.polylines(img, [_fixed_point(apply_h(h_cam, box_a))], True, _COLOR_A, t, cv2.LINE_AA, shift=4)
    _dashed(img, apply_h(h_cam, box_b), _COLOR_B, t, dash=12.0 * t)
    _dashed(img, apply_h(h_cam, rect_polygon(*scene.setup.content_rect_mm)), _COLOR_CONTENT, max(1, t // 2),
            dash=6.0 * t)

    scale = img.shape[0] * 0.035 / cv2.getTextSize("H", cv2.FONT_HERSHEY_SIMPLEX, 1.0, 1)[0][1]
    pts, masks = regions or region_masks(scene)
    centre = tuple(np.vstack([box_a, box_b]).mean(axis=0))
    for text, region, color in (("A only", "only_a", _COLOR_A), ("B only", "only_b", _COLOR_B),
                                ("overlap", "overlap", _COLOR_OVERLAP)):
        found = deepest_point(pts, masks[region], centre)
        if found is None:
            continue
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, t)
        u, v = apply_h(h_cam, found[0])
        cv2.putText(img, text, (int(u - tw / 2), int(v + th / 2)), cv2.FONT_HERSHEY_SIMPLEX, scale, color, t,
                    cv2.LINE_AA)
    return img


def caption(width: int, text: str) -> np.ndarray:
    height = max(28, width // 80)
    strip = np.full((height, width, 3), 32, dtype=np.uint8)
    scale = 0.55 * height / 22
    cv2.putText(strip, text, (height // 3, int(height * 0.68)), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (235, 235, 235), max(1, round(scale * 1.3)), cv2.LINE_AA)
    return strip


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="output directory (default out/<scenario>)")
    parser.add_argument("--quality", choices=sorted(QUALITY), default=None)
    args = parser.parse_args(argv)

    scene = load_scene(args.scenario, quality=args.quality)
    out_dir = args.out or Path("out") / scene.name
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    renderer = scene.renderer()
    content = scene.content_image()
    t1 = time.perf_counter()
    result = renderer.render(content, scene.frame_rng(0))
    t2 = time.perf_counter()

    electrons = scene.camera.decode(result.frame)
    black = scene.projectors[scene.setup.names[0]].black_level
    view = log_display(electrons / np.float32(white_electrons(scene)), black)
    text = (f"{scene.name}  |  frame 0  |  log display y = ln(1+x/b)/ln(1+1/b), x = radiance / one projector's white,"
            f" b = black level = 1/{1 / black:.0f}")
    legend = ("true geometry: orange = projector A's box (lit raster)  |  blue dashed = projector B's box  |"
              "  green = overlap (A and B both shine; blended)  |  white dashed = content rect")
    regions = region_masks(scene)
    view = np.vstack([view, caption(view.shape[1], text), overlay_panel(scene, view, regions),
                      caption(view.shape[1], legend)])
    path = out_dir / "view.png"
    cv2.imwrite(str(path), cv2.cvtColor(view, cv2.COLOR_RGB2BGR))

    summary: dict = {
        "scenario": scene.name,
        "view": str(path),
        "frame": list(scene.camera.resolution),
        "screen_grid": list(result.grid.shape),
        "setup_s": round(t1 - t0, 3),
        "render_s": round(t2 - t1, 3),
    }
    levels = measure_levels(scene, electrons, regions)
    white = white_electrons(scene)
    summary["levels"] = levels
    summary["levels_display_255"] = {
        k: None if v is None else int(log_display(np.array(v["mean_e"] / white), black)) for k, v in levels.items()
    }
    print(json.dumps(summary))
    return summary


if __name__ == "__main__":
    main()
