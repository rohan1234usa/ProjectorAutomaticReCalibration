"""Gallery figures: the six arrangements, the nuisances, the gain screen and the zoomed camera.

Arrangements (CLAUDE.md decision 2): the projectors may overlap in any convex shape. Each preset
is shown as the camera sees it, with its blend map -- who lights each point of the screen -- and
its calibrated geometry for the boundary toy on the algorithm page.

Nuisances (decision 12) change the picture without moving either projector, so truth stays
aligned and the detector must keep answering NO. Each is shown before and during, with what
changed between the two. "Before" sometimes comes from the sweep's quietest variant (the one
with only a person passing, outside that person's window), because flicker and sharpening are
on from the first frame and the lamp's slide changes during its dimming ramp. Each caption is
written from the nuisance's own parameters in the scenario file.

A gain screen sends more light back in one direction, so each projector has a hotspot; a matte
screen has none. The zoomed camera frames the overlap only, sensor turned 90° for a vertical
overlap: its frames are turned back here so screen x runs left to right.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from demo import images as im
from demo.figures import SMALL, blend_map, display, frame, public, px_per_mm, save, textured_windows
from demo.manifest import Shot
from demo.overlays import geometry, layers, piece_counts
from demo.renders import Family
from sim.nuisance import Nuisances
from sim.planar import apply_h, clip_convex
from sim.room import ScreenGain


def arrangement(out: Path, fam: Family, preset: str, i: int, edge_piece_mm: float) -> tuple[dict[str, Any], dict[str, Any]]:
    """(the samples page's figure, the boundary toy's geometry) for one arrangement preset."""
    fig = frame(out, fam, f"arrangement_{preset}", fam.base, i, width=SMALL)
    shape = geometry(preset, fam.scene.setup, fam.scene.screen.size_mm, edge_piece_mm)
    fig.update({"preset": preset, "blend": blend_map(out, fam, f"blend_{preset}"), "pieces": piece_counts(shape["pieces"])})
    return fig, shape


def capture(fam: Family, variant: str, i: int) -> dict[str, Any]:
    """A frame kept in memory, so it can be compared after its renderer has been released."""
    facts = fam.facts(variant, i)
    return {"e": fam.electrons(variant, i), "facts": public(facts),
            "overlap_px": apply_h(facts["camera_h"], clip_convex(facts["boxes_mm"]["a"], facts["boxes_mm"]["b"]))}


def change_map(out: Path, fam: Family, name: str, before: np.ndarray, during: np.ndarray) -> tuple[dict[str, Any], float]:
    """Where two frames differ, at full colour from the 99.9th percentile of the change; and that scale."""
    diff = np.abs(during - before)
    scale = max(float(np.percentile(diff, 99.9)), 0.01 * fam.white)
    dim = (display(fam, before, "natural") * 0.35).astype(np.uint8)
    alpha = np.sqrt(np.clip(im.shrink_max(diff, 2) / scale, 0.0, 1.0))
    return save(out, name, im.tint(im.shrink(dim, dim.shape[1] // 2), alpha, (134, 182, 239))), scale


def pair(out: Path, fam: Family, key: str, before: dict[str, Any], during: dict[str, Any],
         crop: bool = False) -> dict[str, Any]:
    """Before and during pictures of a nuisance, and a map of what changed."""
    diff, scale = change_map(out, fam, f"nuisance_{key}_diff", before["e"], during["e"])
    fig = {"before": save(out, f"nuisance_{key}_before", im.shrink(display(fam, before["e"], "natural"), SMALL)),
           "during": save(out, f"nuisance_{key}_during", im.shrink(display(fam, during["e"], "natural"), SMALL)),
           "facts": during["facts"], "diff": diff, "diff_scale_rel": round(scale / fam.white, 4)}
    if crop:
        window = textured_windows(before["e"], during["overlap_px"], 1)[0]
        fig["crops"] = [save(out, f"nuisance_{key}_crop_{name}", im.natural(im.crop(x["e"], window), fam.white), "png")
                        for name, x in (("before", before), ("during", during))]
    return fig


def caption(nuisances: Nuisances, ambient: float, hidden: int) -> str:
    """What the nuisances of a scenario variant do, with their numbers (`hidden`: markers hidden in the frame shown)."""
    parts = []
    for bump in nuisances.bumps:
        sx, sy = bump.shift_px
        parts.append(f"the camera is knocked {math.hypot(sx, sy):.1f} px ({sx:+g}, {sy:+g}) and turned "
                     f"{bump.rotation_deg:g}° at {float(bump.schedule.onset):g} s")
    for lamp in nuisances.lamps:
        ramp = float(lamp.schedule.duration)
        parts.append(f"projector {lamp.projector.upper()}'s lamp fades to {100 * lamp.gain:g}% "
                     f"{f'over {ramp / 60:g} minutes ' if ramp else ''}from {float(lamp.schedule.onset):g} s")
    for room in nuisances.rooms:
        parts.append(f"the room light steps from {ambient:g} to {room.ambient:g} of projector white at "
                     f"{float(room.schedule.onset):g} s")
    for person in nuisances.occluders:
        parts.append(f"a person crosses in front of the screen for {float(person.duration):g} s from {float(person.t0):g} s")
    for f in nuisances.flickers:
        parts.append(f"projector {f.projector.upper()} flickers by ±{100 * f.amplitude:g}% at {f.frequency_hz:g} Hz, "
                     f"read out {1e6 * f.line_time_s:g} µs per camera row")
    if nuisances.sharpening is not None:
        s = nuisances.sharpening
        parts.append(f"in-camera sharpening (amount {s.amount:g}, σ {s.sigma_px:g} px) is left on")
    text = "; ".join(parts)
    text = text[:1].upper() + text[1:] + "."
    return text + (f" Here {hidden} marker{'s are' if hidden > 1 else ' is'} hidden." if hidden else "")


def hotspot_point(fam: Family, name: str) -> np.ndarray | None:
    """Where a specular screen would show projector `name`'s hotspot (also on a matte screen, for comparison)."""
    room = fam.scene.room
    probe = dataclasses.replace(room, gain=dataclasses.replace(room.gain or ScreenGain(), peak=2.0))
    return probe.hotspot_mm(name)


def gain(out: Path, fam: Family, name: str, i: int, img: dict[str, Any] | None = None) -> dict[str, Any]:
    """A flat grey frame and its brightness along the row through the hotspots, net of room light.

    `img`: a picture of this very frame already saved (the content gallery's flat grey), reused.
    """
    e = fam.electrons(fam.base, i)
    if img is None:
        fig = frame(out, fam, name, fam.base, i, width=SMALL, e=e)
    else:
        facts = fam.facts(fam.base, i)
        fig = {"img": img, "facts": public(facts), "layers": layers(fam, facts)}
    room = fam.room(fam.base, i)
    spots = [p for p in (hotspot_point(fam, n) for n in fam.scene.setup.names) if p is not None]
    y = float(np.mean([p[1] for p in spots])) if spots else fam.scene.screen.size_mm[1] / 2
    xs = np.arange(0.0, fam.scene.screen.size_mm[0] + 0.1, 5.0)
    ys = np.arange(y - 20.0, y + 20.1, 2.0)
    net = im.sample_mm((e - room) / np.float32(fam.white), fam.facts(fam.base, i)["camera_h"], xs, ys).mean(axis=0)
    fig["profile"] = {"x_mm": xs.round(1).tolist(), "rel": np.round(net, 5).tolist(), "y_mm": round(y, 1)}
    fig["peak"] = fam.scene.room.gain.peak if fam.scene.room.gain is not None else 1.0
    return fig


def zoomed(out: Path, fam: Family, shots: tuple[Shot, ...]) -> dict[str, Any]:
    """The zoomed camera's frames turned upright, and the same text window at each shift."""
    frames, crops = [], []
    window = None
    for shot in shots:
        e = cv2.rotate(np.asarray(fam.electrons(shot.variant, shot.frame)), cv2.ROTATE_90_CLOCKWISE)
        facts = fam.facts(shot.variant, shot.frame)
        size = f"{facts['offset_px']:.3g}"
        lay = layers(fam, facts, labels=False, rotate=True)
        if window is None:
            window = textured_windows(e, np.array(lay["overlap"]), 1)[0]
        view = save(out, f"zoomed_{size}", im.shrink(display(fam, e, "natural"), 520))
        frames.append({"size_px": float(size), "img": view, "layers": lay, "facts": public(facts)})
        crop = im.natural(im.crop(e, window), fam.white)
        crops.append({"size_px": float(size), **save(out, f"zoomed_{size}_crop", crop, "png")})
    centre = fam.scene.setup.overlap().mean(axis=0)
    return {"frames": frames, "crops": crops, "px_per_mm": round(px_per_mm(fam.scene.camera.h_mm_to_px, centre), 3)}
