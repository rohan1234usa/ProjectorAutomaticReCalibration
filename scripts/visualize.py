"""Quick-look image of one simulated camera frame, for humans.

Phase 1 renders frame 0 of a scenario and writes ``view.png``: the camera frame shown with a
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
frame -- the unlit screen, each projector's black, and the overlap's doubled black. Patch
positions come from the scenario's true geometry, which is fine here: this is harness code,
not the detector.

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

from sim.planar import apply_h, clip_convex, rect_polygon
from sim.scenario import Scene, load_scene

# Overlay colours (RGB).
_COLOR_A = (255, 150, 30)
_COLOR_B = (60, 170, 255)
_COLOR_OVERLAP = (60, 220, 90)
_COLOR_CONTENT = (235, 235, 235)


def log_display(x: np.ndarray, black: float) -> np.ndarray:
    """Map relative radiance to 8-bit display values with the log curve above."""
    y = np.log1p(np.maximum(x, 0.0) / black) / math.log1p(1.0 / black)
    return np.clip(np.rint(255.0 * y), 0, 255).astype(np.uint8)


def white_electrons(scene: Scene) -> float:
    """Electrons recorded (image centre) for one projector's full white on the screen."""
    brightness = scene.projectors[scene.setup.names[0]].brightness
    return scene.screen.reflectance * brightness * scene.camera.exposure * scene.camera.full_well_e


def patch_centres_mm(scene: Scene) -> dict[str, tuple[float, float]]:
    """Screen points in the black border (and outside the rasters) where levels are measured.

    Only meaningful for side-by-side scenes whose content has a black border.
    """
    a, b = scene.setup.names
    box_a, box_b = scene.setup.box_mm(a), scene.setup.box_mm(b)
    x0, y0, x1, y1 = scene.setup.content_rect_mm
    hc = scene.setup.content_size()[1]
    border_mm = round(float(scene.content.get("border_frac", 0.0)) * hc) * (y1 - y0) / hc
    y_mid = (y0 + y1) / 2
    return {
        "unlit": (box_a[:, 0].min() / 2, y_mid),
        "black_a": (x0 + border_mm / 2, y_mid),
        "black_b": (x1 - border_mm / 2, y_mid),
        "black_overlap": ((box_a[:, 0].max() + box_b[:, 0].min()) / 2, y0 + border_mm / 2),
    }


def measure_levels(scene: Scene, electrons: np.ndarray) -> dict[str, dict[str, float]]:
    """Mean and per-pixel spread (electrons, all channels) in a small square at each patch."""
    hc = scene.setup.content_size()[1]
    x0, y0, x1, y1 = scene.setup.content_rect_mm
    border_mm = round(float(scene.content.get("border_frac", 0.0)) * hc) * (y1 - y0) / hc
    half = max(2, int(0.3 * border_mm * scene.camera.px_per_mm_at_centre()))
    out = {}
    for name, centre in patch_centres_mm(scene).items():
        u, v = np.rint(apply_h(scene.camera.h_mm_to_px, np.array(centre))).astype(int)
        patch = electrons[v - half : v + half + 1, u - half : u + half + 1]
        out[name] = {"mean_e": round(float(patch.mean()), 2), "std_e": round(float(patch.std()), 2)}
    return out


def _dashed(img: np.ndarray, poly_px: np.ndarray, color: tuple[int, int, int], thickness: int, dash: float) -> None:
    """Closed dashed outline, so a second outline lying on top of another stays distinguishable."""
    for p, q in zip(poly_px, np.roll(poly_px, -1, axis=0)):
        length = float(np.hypot(*(q - p)))
        for s in np.arange(0.0, length, 2 * dash):
            a = p + (q - p) * (s / length)
            b = p + (q - p) * (min(s + dash, length) / length)
            cv2.line(img, tuple(np.rint(a * 16).astype(int)), tuple(np.rint(b * 16).astype(int)),
                     color, thickness, cv2.LINE_AA, shift=4)


def overlay_panel(scene: Scene, view: np.ndarray) -> np.ndarray:
    """The frame dimmed to gray, with box A, box B, their overlap and the content rect drawn on it."""
    img = np.repeat((view.mean(axis=2, keepdims=True) * 0.55).astype(np.uint8), 3, axis=2)
    a, b = scene.setup.names
    box_a, box_b = scene.setup.box_mm(a), scene.setup.box_mm(b)
    to_px = lambda poly_mm: apply_h(scene.camera.h_mm_to_px, poly_mm)  # noqa: E731
    fixed = lambda poly_px: np.rint(poly_px * 16).astype(np.int32)  # noqa: E731  (4 fractional bits)
    overlap_px = to_px(clip_convex(box_a, box_b))
    t = max(2, img.shape[1] // 900)

    fill = img.copy()
    cv2.fillPoly(fill, [fixed(overlap_px)], _COLOR_OVERLAP, cv2.LINE_AA, shift=4)
    img = cv2.addWeighted(fill, 0.35, img, 0.65, 0.0)
    cv2.polylines(img, [fixed(overlap_px)], True, _COLOR_OVERLAP, max(1, t // 2), cv2.LINE_AA, shift=4)
    cv2.polylines(img, [fixed(to_px(box_a))], True, _COLOR_A, t, cv2.LINE_AA, shift=4)
    _dashed(img, to_px(box_b), _COLOR_B, t, dash=12.0 * t)
    _dashed(img, to_px(rect_polygon(*scene.setup.content_rect_mm)), _COLOR_CONTENT, max(1, t // 2), dash=6.0 * t)

    height = img.shape[0] * 0.035
    scale = height / cv2.getTextSize("H", cv2.FONT_HERSHEY_SIMPLEX, 1.0, 1)[0][1]
    ov_x = (box_a[:, 0].max() + box_b[:, 0].min()) / 2
    y_mid = (box_a[:, 1].mean() + box_b[:, 1].mean()) / 2
    labels = (
        ("A only", ((box_a[:, 0].min() + box_b[:, 0].min()) / 2, y_mid), _COLOR_A),
        ("B only", ((box_a[:, 0].max() + box_b[:, 0].max()) / 2, y_mid), _COLOR_B),
        ("overlap", (ov_x, y_mid), _COLOR_OVERLAP),
    )
    for text, centre_mm, color in labels:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, max(1, t))
        u, v = apply_h(scene.camera.h_mm_to_px, np.array(centre_mm))
        cv2.putText(img, text, (int(u - tw / 2), int(v + th / 2)), cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                    max(1, t), cv2.LINE_AA)
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
    parser.add_argument("--quality", choices=["fast", "standard"], default=None)
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
    view = np.vstack([view, caption(view.shape[1], text), overlay_panel(scene, view), caption(view.shape[1], legend)])
    path = out_dir / "view.png"
    cv2.imwrite(str(path), cv2.cvtColor(view, cv2.COLOR_RGB2BGR))

    summary = {
        "scenario": scene.name,
        "view": str(path),
        "frame": list(scene.camera.resolution),
        "screen_grid": list(result.grid.shape),
        "setup_s": round(t1 - t0, 3),
        "render_s": round(t2 - t1, 3),
    }
    if float(scene.content.get("border_frac", 0.0)) > 0:
        levels = measure_levels(scene, electrons)
        white = white_electrons(scene)
        summary["levels"] = levels
        summary["levels_display_255"] = {
            k: int(log_display(np.array(v["mean_e"] / white), black)) for k, v in levels.items()
        }
    print(json.dumps(summary))
    return summary


if __name__ == "__main__":
    main()
