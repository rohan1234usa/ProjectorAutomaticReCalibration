"""Sample-data figures: what the simulated camera records, and the truth behind each frame.

Every picture here is a real simulator frame (CLAUDE.md section 5) at the quality the build asks
for, shrunk for the web. Each comes with its ground truth -- the true offset between the
projectors, where each box lands -- which the detector will never see; the page draws it on
top so a reader can tell projector edges from content edges.

The misalignment series uses the sweeps' pairing: every variant of ``shift_sweep`` shares its
content and its noise, so subtracting the aligned twin's frame leaves only what B's move
changed. Where A shines alone the difference is exactly zero, which the figure reports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from demo import images as im
from demo.manifest import SHOTS, SIZES, TWIN, across
from demo.overlays import layers
from demo.renders import Family
from sim.dataset import frame_hash
from sim.planar import apply_h, clip_convex, signed_area

WIDE = 1600  # whole frames on the pages
SMALL = 960  # gallery frames
CROP = 260  # overlap crops, camera px (shown enlarged)


def save(out: Path, name: str, img: np.ndarray, ext: str = "jpg") -> dict[str, Any]:
    name = name.replace(".", "p")
    return {"src": f"img/{name}.{ext}", **im.save(out / "img" / f"{name}.{ext}", img)}


def display(fam: Family, e: np.ndarray, mode: str) -> np.ndarray:
    return im.log(e, fam.white, fam.black) if mode == "log" else im.natural(e, fam.white)


def public(facts: dict[str, Any]) -> dict[str, Any]:
    """The facts a caption shows (numbers rounded, no matrices)."""
    return {"variant": facts["variant"], "frame": facts["frame"], "t_s": facts["t_s"], "tag": facts["tag"],
            "offset_mm": round(facts["offset_mm"], 4), "offset_px": round(facts["offset_px"], 4),
            "aligned": facts["aligned"], "ambient": facts["ambient"],
            "lamp": {k: round(v, 4) for k, v in facts["lamp"].items()}, "camera_bump": facts["camera_bump"],
            "people": facts["people"], "markers_visible": len(facts["markers_visible"])}


def mask(fam: Family, poly_mm: np.ndarray, camera_h: np.ndarray, erode: int = 0) -> np.ndarray:
    """Camera pixels inside a screen polygon (optionally shrunk by `erode` px)."""
    w, h = fam.scene.camera.resolution
    m = np.zeros((h, w), np.uint8)
    if len(poly_mm) >= 3:
        cv2.fillPoly(m, [np.rint(apply_h(camera_h, poly_mm) * 16).astype(np.int32)], 1, cv2.LINE_8, shift=4)
    if erode:
        m = cv2.erode(m, np.ones((2 * erode + 1, 2 * erode + 1), np.uint8))
    return m.astype(bool)


def frame(out: Path, fam: Family, name: str, variant: str, i: int, mode: str = "natural", width: int = WIDE,
          log_too: bool = False, e: np.ndarray | None = None) -> dict[str, Any]:
    """One frame: picture(s), its facts, and its true geometry for the overlay."""
    e = fam.electrons(variant, i) if e is None else e
    facts = fam.facts(variant, i)
    fig = {"img": save(out, name, im.shrink(display(fam, e, mode), width)), "facts": public(facts),
           "layers": layers(fam, facts), "mode": mode}
    if log_too:
        fig["img_log"] = save(out, name + "_log", im.shrink(display(fam, e, "log"), width))
    return fig


def textured_windows(e: np.ndarray, overlap_px: np.ndarray, count: int = 2) -> list[tuple[int, int, int, int]]:
    """The `count` most textured CROP-sized windows centred on the overlap, not overlapping each other."""
    u_mid = int(round(overlap_px[:, 0].mean()))
    v_lo, v_hi = int(overlap_px[:, 1].min()) + 8, int(overlap_px[:, 1].max()) - 8 - CROP
    lap = np.abs(cv2.Laplacian(e.astype(np.float32), cv2.CV_32F))
    scored = sorted(((float(lap[v : v + CROP, u_mid - CROP // 2 : u_mid + CROP // 2].mean()), v)
                     for v in range(v_lo, v_hi, 10)), reverse=True)
    chosen: list[int] = []
    for _, v in scored:
        if all(abs(v - c) >= CROP for c in chosen):
            chosen.append(v)
        if len(chosen) == count:
            break
    return [(u_mid - CROP // 2, v, u_mid + CROP // 2, v + CROP) for v in sorted(chosen)]


def shift_series(out: Path, fam: Family) -> dict[str, Any]:
    """From invisible to obvious: B moved 0 ... 8 px across, the same slide, the same noise."""
    i = SHOTS["shift_0"].frame
    twin = fam.electrons(TWIN, i)
    base = fam.facts(TWIN, i)
    overlap_px = apply_h(base["camera_h"], clip_convex(base["boxes_mm"]["a"], base["boxes_mm"]["b"]))
    windows = textured_windows(twin, overlap_px)
    scale = 0.25 * fam.white  # the difference that shows at full colour
    dim = (im.natural(twin, fam.white) * 0.35).astype(np.uint8)
    dim = im.shrink(dim, dim.shape[1] // 2)

    def step(key: str, variant: str) -> dict[str, Any]:
        e = twin if variant == TWIN else fam.electrons(variant, i)
        facts = fam.facts(variant, i)
        diff = np.abs(e - twin)
        b_either = mask(fam, base["boxes_mm"]["b"], base["camera_h"]) | mask(fam, facts["boxes_mm"]["b"], facts["camera_h"])
        b_either = cv2.dilate(b_either.astype(np.uint8), np.ones((21, 21), np.uint8)).astype(bool)  # and its blur
        a_only = mask(fam, facts["boxes_mm"]["a"], facts["camera_h"], 3) & ~b_either  # A alone in both frames
        alpha = np.sqrt(np.clip(im.shrink_max(diff, 2) / scale, 0.0, 1.0))
        view = im.natural(e, fam.white)
        return {"variant": variant, "offset_mm": round(facts["offset_mm"], 4),
                "offset_px": round(facts["offset_px"], 4),
                "crops": [save(out, f"{key}_crop{n}", im.crop(view, w), "png") for n, w in enumerate(windows)],
                "diff": save(out, f"{key}_diff", im.tint(dim, alpha, (134, 182, 239))),
                "a_only_max_e": round(float(diff[a_only].max()), 3), "max_e": round(float(diff.max()), 1)}

    steps = [{"size_px": float(s), **step(f"shift_{s}", across(s) if s != "0" else TWIN)} for s in SIZES]
    along = {"size_px": 8.0, **step("shift_along_8", SHOTS["shift_along_8"].variant)}
    return {"frame": i, "t_s": base["t_s"], "steps": steps, "along": along, "windows": windows,
            "diff_scale_e": round(scale, 1), "white_e": round(fam.white, 1), "overlap_px": overlap_px.round(1).tolist(),
            "frame_size": list(fam.scene.camera.resolution)}


def blend_map(out: Path, fam: Family, name: str, step_mm: float = 4.0) -> dict[str, Any]:
    """Who lights each point: A's orange and B's blue mixed by their blend weights (a + b = 1 in the overlap)."""
    setup, (sw, sh) = fam.scene.setup, fam.scene.screen.size_mm
    xs, ys = np.arange(step_mm / 2, sw, step_mm), np.arange(step_mm / 2, sh, step_mm)
    pts = np.stack(np.meshgrid(xs, ys), axis=-1)
    w = setup.blend_at(pts)
    img = np.full((*w["a"].shape, 3), 13.0)
    lit = (w["a"] + w["b"]) > 0
    img[lit] = w["a"][lit, None] * np.float64(im.A) + w["b"][lit, None] * np.float64(im.B)
    img = np.rint(img).astype(np.uint8)
    overlap = setup.overlap()
    return {**save(out, name, img, "png"), "screen_mm": [float(sw), float(sh)],
            "overlap_m2": round(abs(signed_area(overlap)) / 1e6, 3), "overlap_vertices": len(overlap)}


def content_item(out: Path, fam: Family, key: str, label: str, intended: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """One kind of content, with the numbers the frame router would look at (CLAUDE.md 4.3).

    The routing shown is the brief's thresholds applied to the overlap: skip when the frame differs
    from the previous one by more than MOTION_LEVEL, dark when its mean, net of the room light, is
    below DARK_LEVEL. Textured versus flat needs per-tile edge density (Phase 3), so content that is
    neither skipped nor dark keeps the kind it was chosen for, or "lit" if it was meant to be dark.
    """
    shot = SHOTS[key]
    i = shot.frame
    e = fam.electrons(shot.variant, i)
    dark = intended == "dark" or shot.tag in ("black", "dark", "video_dark")
    fig = frame(out, fam, f"content_{key}", shot.variant, i, "log" if dark else "natural", SMALL, e=e)
    facts = fam.facts(shot.variant, i)
    inside = mask(fam, clip_convex(facts["boxes_mm"]["a"], facts["boxes_mm"]["b"]), facts["camera_h"], 4)
    room = fam.room(shot.variant, i)
    fig.update({"key": key, "label": label, "intended": intended,
                "overlap_mean": round(float(e[inside].mean() / fam.white), 5),
                "overlap_mean_net": round(float((e - room)[inside].mean() / fam.white), 5)})
    if intended == "skip" or shot.tag == "video_dark":
        before = fam.electrons(shot.variant, i - 1)
        fig["motion"] = round(float(np.abs(e - before).mean() / fam.white), 5)
    if fig.get("motion", 0.0) > cfg["MOTION_LEVEL"]:
        fig["routing"] = "skip"
    elif fig["overlap_mean_net"] < cfg["DARK_LEVEL"]:
        fig["routing"] = "dark"
    else:
        fig["routing"] = intended if intended in ("textured", "flat") else "lit"
    return fig


def inputs(fam: Family) -> dict[str, Any]:
    """What the detector may read (setup.json) next to what it never sees (one metadata line).

    The line is the one ``make_dataset`` writes for the 8 px variant, so the family must render that
    variant itself: FrameSource.truth describes its own scenario only.
    """
    shot = SHOTS["shift_8"]
    if fam.base != shot.variant:
        raise ValueError(f"inputs: the family renders {fam.base}, the metadata line needs {shot.variant}")
    line = fam.source.truth(shot.frame)
    line["frame_sha256"], line["png"] = frame_hash(fam.frame(fam.base, shot.frame)), None
    return {"setup": fam.source.setup_dict(), "line": line, "variant": fam.base, "frame": shot.frame}
