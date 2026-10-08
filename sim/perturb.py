"""Misalignment injection: where each projector's pixels really land after it has moved.

A projector that shifts, turns or tilts on its mount changes where its pixels land, while the
calibration software keeps sending a framebuffer built for the old geometry. We model the move
as a change of the screen-side geometry, h_actual = M(t) h_cal, with M a homography in screen
millimetres acting on projector A, B or both:

  shift     M = T(d): every pixel lands d mm further along one direction;
  rotation  M = T(c) R(theta) T(-c): turned about a pivot c (theta > 0 turns +x toward +y);
  scale     M = T(c) S(s) T(-c): zoomed about c;
  keystone  M = T(c) K T(-c), K = [[1,0,0],[0,1,0],[k vx, k vy, 1]]: tilted. For k > 0 pixels on
            the +v side of the pivot crowd together and those on the -v side spread out.

Directions follow the overlap. ``across`` is the length-weighted normal of the moving
projector's inner edges (where it fades out inside its partner), pointing away from the
partner: a positive shift across narrows the overlap and darkens it. ``along`` is ``across``
turned by +90 degrees. Pivots: ``centre`` (where the raster's centre pixel lands),
``far_corner`` (the box corner farthest from the overlap's centroid), ``overlap_centre``, or a
point [x, y] in mm.

Sizes can be given as the offset they cause, ``magnitude_px`` (coarse projector pixels) or
``magnitude_mm``, and the parameter is solved for it: for a shift d = magnitude; for a rotation
2 R sin(theta / 2) = magnitude and for a scale |s - 1| R = magnitude, with R the farthest
overlap vertex from the pivot; for a keystone k by bisection on the true offset. Rotations and
scales may instead give ``deg`` or ``factor`` directly.

The schedule (``sim/schedule.py``) scales the parameter over time, m(t) x parameter. Every
perturbation must name one (``{type: none}`` switches it off), so a forgotten schedule cannot
silently leave a run aligned. For continuous schedules m is quantized so that the offset moves in
steps of 0.02 px: the renderer then sees a finite set of geometries and its caches keep working.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import numpy as np

from sim.calibration import CalibrationSetup
from sim.cfg import check_keys, choice, num, pair
from sim.planar import apply_h, box_mm, centroid, translation
from sim.schedule import Schedule
from sim.schedule import from_config as schedule_from_config
from sim.truth import coarse_pitch_mm, max_separation

KINDS = ("shift", "rotation", "scale", "keystone")
QUANTUM_PX = 0.02  # displacement step of continuous schedules
_SIZES = {"magnitude_px", "magnitude_mm", "deg", "factor"}
_KEYS = {  # keys each kind understands; anything else is a mistake worth reporting
    "shift": {"kind", "schedule", "direction"} | _SIZES,
    "rotation": {"kind", "schedule", "pivot"} | _SIZES,
    "scale": {"kind", "schedule", "pivot"} | _SIZES,
    "keystone": {"kind", "schedule", "pivot", "axis"} | _SIZES,
}


@dataclass(frozen=True, eq=False)
class Perturbation:
    projector: str
    kind: str
    parameter: float  # at m = 1 -- shift: mm; rotation: radians; scale: s - 1; keystone: k (1/mm)
    vector: np.ndarray  # shift direction or keystone axis (unit); unused otherwise
    pivot: np.ndarray  # mm
    schedule: Schedule
    quantum: float | None  # step of m for continuous schedules
    spec: dict[str, Any] = field(default_factory=dict)  # as written, for the dataset's tags

    def multiplier(self, t: Fraction) -> float:
        m = self.schedule.value(t)
        if self.quantum:
            m = round(m / self.quantum) * self.quantum
        return m

    def matrix(self, m: float) -> np.ndarray:
        return transform(self.kind, m * self.parameter, self.vector, self.pivot)

    def tag(self, m: float) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": self.kind, "multiplier": m}
        if self.kind == "shift":
            out["vector_mm"] = [float(v) for v in m * self.parameter * self.vector]
        elif self.kind == "rotation":
            out["deg"] = math.degrees(m * self.parameter)
        elif self.kind == "scale":
            out["factor"] = 1.0 + m * self.parameter
        else:
            out["k_per_mm"] = m * self.parameter
        return out


def transform(kind: str, value: float, vector: np.ndarray, pivot: np.ndarray) -> np.ndarray:
    """The screen-mm homography of one perturbation at parameter `value`."""
    if kind == "shift":
        return translation(*(value * vector))
    if kind == "rotation":
        c, s = math.cos(value), math.sin(value)
        local = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    elif kind == "scale":
        local = np.diag([1.0 + value, 1.0 + value, 1.0])
    else:
        local = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [value * vector[0], value * vector[1], 1.0]])
    return translation(*pivot) @ local @ translation(*(-pivot))


def h_actual(
    setup: CalibrationSetup, perturbations: tuple[Perturbation, ...], multipliers: tuple[float, ...]
) -> dict[str, np.ndarray]:
    """Actual geometry of each projector: its perturbations composed in list order, then h_cal."""
    out = {n: np.asarray(setup.h_cal[n], dtype=np.float64) for n in setup.names}
    for p, m in zip(perturbations, multipliers):
        if m != 0.0:
            out[p.projector] = p.matrix(m) @ out[p.projector]
    return out


def across(setup: CalibrationSetup, name: str) -> np.ndarray:
    """Unit normal of projector `name`'s inner edges, pointing away from its partner."""
    inner = setup.inner_edges()[name]
    if not inner:
        raise ValueError(f"perturbation: projector {name} has no inner edge, so across/along are undefined; give a vector")
    inside = box_mm(setup.h_cal[name], setup.resolution[name]).mean(axis=0)
    total, length = np.zeros(2), 0.0
    for p, q in inner:
        e = q - p
        n = np.array([e[1], -e[0]])  # length |e|
        if n @ ((p + q) / 2 - inside) < 0:
            n = -n  # outward from the box: into the partner
        total += n
        length += float(np.hypot(*e))
    if np.hypot(*total) < 1e-6 * length:  # e.g. a box nested inside its partner: no single direction
        raise ValueError(f"perturbation: projector {name}'s inner edges face every way, so across/along "
                         "are undefined; give a vector")
    return -total / np.hypot(*total)


def pivot_point(setup: CalibrationSetup, name: str, spec: Any) -> np.ndarray:
    if spec == "centre":
        w, h = setup.resolution[name]
        return apply_h(setup.h_cal[name], np.array([(w - 1) / 2, (h - 1) / 2]))
    if spec == "overlap_centre":
        return centroid(setup.overlap())
    if spec == "far_corner":
        corners = box_mm(setup.h_cal[name], setup.resolution[name])  # TL, TR, BR, BL
        d = np.hypot(*(corners - centroid(setup.overlap())).T)
        return corners[int(np.nonzero(d >= d.max() - 1e-9)[0][0])]
    return np.array(pair(spec, "perturbation.pivot"))


def _unit(v: Any, where: str) -> np.ndarray:
    x, y = pair(v, where)
    n = math.hypot(x, y)
    if n == 0:
        raise ValueError(f"{where}: vector must be non-zero")
    return np.array([x, y]) / n


def _solve_keystone(setup: CalibrationSetup, name: str, axis: np.ndarray, pivot: np.ndarray, target_mm: float) -> float:
    """k with offset(keystone k about pivot) = |target_mm|, k of target_mm's sign, by bisection.

    A keystone is not symmetric in k (+k crowds one side, -k the other), so the search runs on
    the requested side rather than solving for |k| and flipping its sign.
    """
    sign = 1.0 if target_mm >= 0 else -1.0
    target_mm = abs(target_mm)
    others = {n: np.eye(3) for n in setup.names}
    poly = setup.overlap()

    def offset(k: float) -> float:
        d = dict(others)
        d[name] = transform("keystone", sign * k, axis, pivot)
        a, b = setup.names
        return max_separation(d[a], d[b], poly)

    r = float(np.hypot(*(poly - pivot).T).max())
    lo, hi = 0.0, target_mm / r**2
    while offset(hi) < target_mm:
        lo, hi = hi, 2 * hi
    while hi - lo > 1e-12 * hi:  # the offset is then within ~1e-9 mm of the target
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if offset(mid) < target_mm else (lo, mid)
    return sign * (lo + hi) / 2


def _one(name: str, spec: Mapping[str, Any], setup: CalibrationSetup, pitch: float, where: str) -> Perturbation:
    if not isinstance(spec, Mapping):
        raise ValueError(f"{where}: expected a mapping like {{kind: shift, magnitude_px: 1}}")
    kind = choice(spec.get("kind"), KINDS, f"{where}.kind")
    check_keys(spec, _KEYS[kind], f"{where} ({kind})")
    sizes = [k for k in ("magnitude_px", "magnitude_mm", "deg", "factor") if k in spec]
    natural = {"rotation": {"deg"}, "scale": {"factor"}}.get(kind, set())
    if len(sizes) != 1 or (sizes[0] in ("deg", "factor") and sizes[0] not in natural):
        allowed = ["magnitude_px", "magnitude_mm", *sorted(natural)]
        raise ValueError(f"{where}: give exactly one size for a {kind}, one of {allowed}")
    size_key = sizes[0]
    target = num(spec[size_key], f"{where}.{size_key}")
    target_mm = target * pitch if size_key == "magnitude_px" else target
    poly = setup.overlap()
    vector, pivot = np.zeros(2), np.zeros(2)
    if kind == "shift":
        direction = spec.get("direction", "across")
        if direction in ("across", "along"):
            a = across(setup, name)
            vector = a if direction == "across" else np.array([-a[1], a[0]])
        else:
            vector = _unit(direction, f"{where}.direction")
        parameter = target_mm
        unit_offset = abs(parameter)
    else:
        pivot = pivot_point(setup, name, spec.get("pivot", "centre"))
        r = float(np.hypot(*(poly - pivot).T).max())  # farthest overlap vertex from the pivot
        if kind == "rotation":
            if size_key == "deg":
                parameter = math.radians(target)
            else:
                parameter = math.copysign(2 * math.asin(min(1.0, abs(target_mm) / (2 * r))), target_mm)
            unit_offset = 2 * r * abs(math.sin(parameter / 2))
        elif kind == "scale":
            parameter = target - 1.0 if size_key == "factor" else target_mm / r
            unit_offset = abs(parameter) * r
        else:
            axis = spec.get("axis", "x")
            if isinstance(axis, str):
                vector = {"x": np.array([1.0, 0.0]), "y": np.array([0.0, 1.0])}[choice(axis, ("x", "y"), f"{where}.axis")]
            else:
                vector = _unit(axis, f"{where}.axis")
            parameter = _solve_keystone(setup, name, vector, pivot, target_mm)
            unit_offset = abs(target_mm)
    if "schedule" not in spec:
        raise ValueError(f"{where}: schedule is required ({{type: step, t0_s: ...}}, or {{type: none}} to switch it off)")
    schedule = schedule_from_config(spec["schedule"], f"{where}.schedule")
    quantum = None
    if schedule.continuous and unit_offset > 0:
        quantum = QUANTUM_PX * pitch / unit_offset  # the step of m that moves the offset by 0.02 px
    return Perturbation(name, kind, parameter, vector, pivot, schedule, quantum, dict(spec))


def from_config(cfg: Any, setup: CalibrationSetup) -> tuple[Perturbation, ...]:
    """Parse ``perturbation: {b: {kind: shift, ...}}`` (a mapping or list per projector), or none."""
    if cfg is None or cfg == "none" or cfg == {}:
        return ()
    check_keys(cfg, set(setup.names), "perturbation")
    pitch = coarse_pitch_mm(setup)
    out = []
    for name in setup.names:  # projectors in a fixed order, perturbations in list order
        specs = cfg.get(name)
        if specs is None:
            continue
        specs = [specs] if isinstance(specs, Mapping) else list(specs)
        out += [_one(name, s, setup, pitch, f"perturbation.{name}[{i}]") for i, s in enumerate(specs)]
    return tuple(out)
