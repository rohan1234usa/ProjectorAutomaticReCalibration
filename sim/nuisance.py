"""Nuisances: things that change the picture without misaligning the projectors.

None of these may ever make the detector answer YES (CLAUDE.md decision 12). Each one changes a
frame's state while its ground truth stays "aligned":

  camera_bump  The camera is knocked: its whole view (screen, bezel, markers) shifts and turns in
               the image from t0 on. The markers move with it, which is how the detector notices
               (FIDUCIAL_MOVE_PX) and re-solves its homography.
  lamp         One projector dims, e.g. to 85% as a lamp ages: all of its light scales together,
               black level included. A projector's light is a scalar weight on its camera
               component, so this costs nothing to render.
  room_light   The room light changes (lights switched on or off): the ambient weight.
  occluder     A person walks across in front of the screen in duration_s, possibly hiding
               markers. Standing near the screen, they are lit by the projector light that would
               have reached the screen behind them, and by the room light, but reflect only about
               30% of it. (Simplification: no cast shadow.)
  sharpening   In-camera sharpening left on by mistake: an unsharp mask on the noisy image, whose
               halos can mimic double contours.
  flicker      Projector brightness modulation beating with the exposure (``sim/flicker.py``).

Camera bumps, lamps and room light follow a schedule (``sim/schedule.py``), which must be given:
``{type: none}`` switches one off, so a forgotten schedule cannot silently do nothing.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import cv2
import numpy as np

from sim import flicker as flicker_module
from sim.cfg import check_keys, choice, num, pair, require, seconds
from sim.flicker import Flicker
from sim.planar import apply_h, translation
from sim.schedule import Schedule
from sim.schedule import from_config as schedule_from_config

_TYPES = ("camera_bump", "lamp", "room_light", "occluder", "sharpening", "flicker")


@dataclass(frozen=True)
class CameraBump:
    shift_px: tuple[float, float]
    rotation_deg: float
    schedule: Schedule

    def image_transform(self, m: float, resolution: tuple[int, int]) -> np.ndarray:
        """How the image moves: turned about the image centre, then shifted (camera pixels)."""
        w, h = resolution
        c = np.array([(w - 1) / 2, (h - 1) / 2])
        t = math.radians(m * self.rotation_deg)
        rot = np.array([[math.cos(t), -math.sin(t), 0.0], [math.sin(t), math.cos(t), 0.0], [0.0, 0.0, 1.0]])
        return translation(m * self.shift_px[0], m * self.shift_px[1]) @ translation(*c) @ rot @ translation(*(-c))


@dataclass(frozen=True)
class Lamp:
    projector: str
    gain: float  # relative light at full effect, e.g. 0.85
    schedule: Schedule

    def value(self, t: Fraction) -> float:
        return 1.0 + (self.gain - 1.0) * self.schedule.value(t)


@dataclass(frozen=True)
class RoomLight:
    ambient: float  # room light at full effect, x one projector's white
    schedule: Schedule


@dataclass(frozen=True)
class Occluder:
    t0: Fraction
    duration: Fraction
    height_mm: float
    floor_mm: float  # screen y of the person's feet
    leftward: bool
    reflectance: float

    def outline(self, t: Fraction, screen_w: float) -> list[np.ndarray] | None:
        """The person's silhouette at t as screen-mm polygons (head, body, legs), or None."""
        if not self.t0 <= t < self.t0 + self.duration:
            return None
        h = self.height_mm
        width = 0.28 * h
        progress = float((t - self.t0) / self.duration)
        x = -width + progress * (screen_w + 2 * width)
        if self.leftward:
            x = screen_w - x
        y = self.floor_mm
        head = np.array([[x + 0.065 * h * math.cos(a), y - 0.87 * h + 0.065 * h * math.sin(a)]
                         for a in np.linspace(0, 2 * math.pi, 16, endpoint=False)])
        body = np.array([[x - 0.14 * h, y - 0.78 * h], [x + 0.14 * h, y - 0.78 * h],
                         [x + 0.11 * h, y - 0.45 * h], [x - 0.11 * h, y - 0.45 * h]])
        legs = np.array([[x - 0.10 * h, y - 0.46 * h], [x + 0.10 * h, y - 0.46 * h], [x + 0.08 * h, y], [x - 0.08 * h, y]])
        return [head, body, legs]


@dataclass(frozen=True)
class Sharpening:
    amount: float
    sigma_px: float

    def apply(self, electrons: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(electrons, (0, 0), self.sigma_px)
        return electrons + np.float32(self.amount) * (electrons - blurred)


@dataclass(frozen=True)
class Nuisances:
    bumps: tuple[CameraBump, ...] = ()
    lamps: tuple[Lamp, ...] = ()
    rooms: tuple[RoomLight, ...] = ()
    occluders: tuple[Occluder, ...] = ()
    sharpening: Sharpening | None = None
    flickers: tuple[Flicker, ...] = ()

    @property
    def onsets(self) -> list[Fraction]:
        out = [x.schedule.onset for x in (*self.bumps, *self.lamps, *self.rooms) if x.schedule.onset is not None]
        return out + [o.t0 for o in self.occluders]


def occluder_mask(polygons: Sequence[np.ndarray], h_mm_to_px: np.ndarray, resolution: tuple[int, int]) -> np.ndarray:
    """Fraction of each camera pixel covered by the silhouette, (h, w) float32 (4x4 supersampled)."""
    w, h = resolution
    k = 4
    mask = np.zeros((h * k, w * k), np.uint8)
    to_sub = np.array([[k, 0.0, (k - 1) / 2], [0.0, k, (k - 1) / 2], [0.0, 0.0, 1.0]])
    for poly in polygons:
        pts = apply_h(to_sub @ h_mm_to_px, poly)
        cv2.fillPoly(mask, [np.rint(pts * 16).astype(np.int32)], 255, cv2.LINE_8, shift=4)
    return cv2.resize(mask.astype(np.float32) / 255.0, (w, h), interpolation=cv2.INTER_AREA)


def from_config(cfg: Any, projectors: tuple[str, ...], screen_size_mm: tuple[float, float], seed: int) -> Nuisances:
    """Parse the scenario's ``nuisances`` list."""
    if not cfg:
        return Nuisances()
    if not isinstance(cfg, list):
        raise ValueError("nuisances: expected a list of {type: ...} mappings")
    found: dict[str, list] = {t: [] for t in _TYPES}
    for i, item in enumerate(cfg):
        where = f"nuisances[{i}]"
        if not isinstance(item, Mapping):
            raise ValueError(f"{where}: expected a mapping")
        kind = choice(item.get("type"), _TYPES, f"{where}.type")
        rest = {k: v for k, v in item.items() if k != "type"}
        if kind in ("camera_bump", "lamp", "room_light"):
            require(rest, {"schedule"}, where)  # {type: none} switches one off; forgetting must not
        if kind == "camera_bump":
            check_keys(rest, {"shift_px", "rotation_deg", "schedule"}, where)
            found[kind].append(CameraBump(pair(rest.get("shift_px", [0, 0]), f"{where}.shift_px"),
                                          num(rest.get("rotation_deg", 0.0), f"{where}.rotation_deg"),
                                          schedule_from_config(rest.get("schedule"), f"{where}.schedule")))
        elif kind == "lamp":
            check_keys(rest, {"projector", "gain", "schedule"}, where)
            found[kind].append(Lamp(choice(rest.get("projector"), projectors, f"{where}.projector"),
                                    num(rest.get("gain", 0.85), f"{where}.gain"),
                                    schedule_from_config(rest.get("schedule"), f"{where}.schedule")))
        elif kind == "room_light":
            check_keys(rest, {"ambient", "schedule"}, where)
            require(rest, {"ambient"}, where)
            found[kind].append(RoomLight(num(rest["ambient"], f"{where}.ambient"),
                                         schedule_from_config(rest.get("schedule"), f"{where}.schedule")))
        elif kind == "occluder":
            check_keys(rest, {"t0_s", "duration_s", "height_mm", "floor_mm", "direction", "reflectance"}, where)
            require(rest, {"t0_s"}, where)
            found[kind].append(Occluder(
                seconds(rest["t0_s"], f"{where}.t0_s"), seconds(rest.get("duration_s", 6), f"{where}.duration_s"),
                num(rest.get("height_mm", 1750.0), f"{where}.height_mm"),
                num(rest.get("floor_mm", screen_size_mm[1] + 400.0), f"{where}.floor_mm"),
                choice(rest.get("direction", "right"), ("left", "right"), f"{where}.direction") == "left",
                num(rest.get("reflectance", 0.3), f"{where}.reflectance")))
        elif kind == "sharpening":
            check_keys(rest, {"amount", "sigma_px"}, where)
            found[kind].append(Sharpening(num(rest.get("amount", 0.8), f"{where}.amount"),
                                          num(rest.get("sigma_px", 1.0), f"{where}.sigma_px")))
        else:
            found[kind].append(flicker_module.from_config(item, projectors, seed, len(found["flicker"]), where))
    if len(found["sharpening"]) > 1:
        raise ValueError("nuisances: at most one sharpening")
    return Nuisances(tuple(found["camera_bump"]), tuple(found["lamp"]), tuple(found["room_light"]),
                     tuple(found["occluder"]), found["sharpening"][0] if found["sharpening"] else None,
                     tuple(found["flicker"]))
