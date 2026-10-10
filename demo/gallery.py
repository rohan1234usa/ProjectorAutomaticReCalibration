"""Gallery figures: the six arrangements, the nuisances, the gain screen and the zoomed camera.

Arrangements (CLAUDE.md decision 2): the projectors may overlap in any convex shape. Each preset
is shown as the camera sees it, with its blend map -- who lights each point of the screen -- and
its calibrated geometry for the boundary toy on the algorithm page.

Nuisances (decision 12) change the picture without moving either projector, so truth stays
aligned and the detector must keep answering NO. Each is shown before and during, with what
changed between the two. "Before" sometimes comes from the sweep's quietest variant (the one
with only a person passing, outside that person's window), because flicker and sharpening are
on from the first frame and the lamp's slide changes during its dimming ramp.

A gain screen sends more light back in one direction, so each projector has a hotspot; a matte
screen has none. The zoomed camera frames the overlap only, sensor turned 90° for a vertical
overlap: its frames are turned back here so screen x runs left to right.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from demo import images as im
from demo.figures import SMALL, blend_map, display, frame, public, save, textured_windows
from demo.manifest import Shot
from demo.overlays import geometry, layers
from demo.renders import Family
from sim.planar import apply_h, clip_convex


def arrangement(out: Path, fam: Family, preset: str, i: int, edge_piece_mm: float) -> dict[str, Any]:
    fig = frame(out, fam, f"arrangement_{preset}", fam.base, i, width=SMALL)
    fig["preset"] = preset
    fig["blend"] = blend_map(out, fam, f"blend_{preset}")
    fig["geometry"] = geometry(fam.scene.setup, fam.scene.screen.size_mm, edge_piece_mm)
    return fig


def capture(fam: Family, variant: str, i: int) -> dict[str, Any]:
    """A frame kept in memory, so it can be compared after its renderer has been released."""
    facts = fam.facts(variant, i)
    return {"e": fam.electrons(variant, i), "facts": public(facts), "layers": layers(fam, facts, labels=False)}


def pair(out: Path, fam: Family, key: str, before: dict[str, Any], during: dict[str, Any],
         crop: bool = False) -> dict[str, Any]:
    """Before and during pictures of a nuisance, and a map of what changed."""
    diff = np.abs(during["e"] - before["e"])
    scale = max(float(np.percentile(diff, 99.9)), 0.01 * fam.white)
    dim = (display(fam, before["e"], "natural") * 0.35).astype(np.uint8)
    alpha = np.sqrt(np.clip(im.shrink_max(diff, 2) / scale, 0.0, 1.0))
    fig = {
        "key": key,
        "before": {"img": save(out, f"nuisance_{key}_before", im.shrink(display(fam, before["e"], "natural"), SMALL)),
                   "facts": before["facts"], "layers": before["layers"]},
        "during": {"img": save(out, f"nuisance_{key}_during", im.shrink(display(fam, during["e"], "natural"), SMALL)),
                   "facts": during["facts"], "layers": during["layers"]},
        "diff": save(out, f"nuisance_{key}_diff", im.tint(im.shrink(dim, dim.shape[1] // 2), alpha, (134, 182, 239))),
        "diff_scale_rel": round(scale / fam.white, 4),
        "mean_change_rel": round(float(diff.mean() / fam.white), 5),
    }
    if crop:
        overlap = np.array(during["layers"]["overlap"])
        window = textured_windows(before["e"], overlap, 1)[0]
        fig["crops"] = [save(out, f"nuisance_{key}_crop_{name}", im.crop(display(fam, x["e"], "natural"), window), "png")
                        for name, x in (("before", before), ("during", during))]
    return fig


def mirror_point(fam: Family, name: str) -> np.ndarray:
    """Where a specular screen would show projector `name`'s hotspot (also on a matte screen, for comparison)."""
    p, c = fam.scene.room.projectors[name], fam.scene.room.camera
    return p[:2] + (c[:2] - p[:2]) * (p[2] / (p[2] + c[2]))


def gain(out: Path, fam: Family, name: str, i: int) -> dict[str, Any]:
    """A flat grey frame and its brightness along the row through the hotspots, net of room light."""
    e = fam.electrons(fam.base, i)
    fig = frame(out, fam, name, fam.base, i, width=SMALL, e=e)
    room = fam.room(fam.base, i)
    y = float(np.mean([mirror_point(fam, n)[1] for n in fam.scene.setup.names]))
    xs = np.arange(0.0, fam.scene.screen.size_mm[0] + 0.1, 5.0)
    ys = np.arange(y - 20.0, y + 20.1, 2.0)
    net = im.sample_mm((e - room) / np.float32(fam.white), fam.facts(fam.base, i)["camera_h"], xs, ys).mean(axis=0)
    fig["profile"] = {"x_mm": xs.round(1).tolist(), "rel": np.round(net, 5).tolist(), "y_mm": round(y, 1)}
    fig["peak"] = fam.scene.room.gain.peak if fam.scene.room.gain is not None else 1.0
    return fig


def zoomed(out: Path, fam: Family, shots: tuple[Shot, ...]) -> dict[str, Any]:
    """The zoomed camera's frames turned upright, and the same text window at each shift."""
    h = fam.scene.camera.resolution[1]
    frames, crops = [], []
    window = None
    for shot in shots:
        e = cv2.rotate(fam.electrons(shot.variant, shot.frame), cv2.ROTATE_90_CLOCKWISE)
        facts = fam.facts(shot.variant, shot.frame)
        size = f"{facts['offset_px']:.3g}"
        view = display(fam, e, "natural")
        lay = layers(fam, facts, labels=False, rotate=True)
        if window is None:
            window = textured_windows(e, np.array(lay["overlap"]), 1)[0]
        frames.append({"size_px": float(size), "img": save(out, f"zoomed_{size}", im.shrink(view, 520)),
                       "layers": lay, "facts": public(facts)})
        crops.append(save(out, f"zoomed_{size}_crop", im.crop(view, window), "png"))
    camera = fam.scene.camera
    overlap = clip_convex(fam.scene.setup.box_mm("a"), fam.scene.setup.box_mm("b"))
    centre = overlap.mean(axis=0)
    step = np.linalg.norm(apply_h(camera.h_mm_to_px, centre + [1.0, 0.0]) - apply_h(camera.h_mm_to_px, centre))
    return {"frames": frames, "crops": crops, "window": list(window), "px_per_mm": round(float(step), 3),
            "rotated_size": [h, camera.resolution[0]]}
