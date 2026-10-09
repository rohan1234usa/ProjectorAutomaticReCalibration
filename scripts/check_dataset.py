"""Check a generated dataset against what the scenario asked for, independently of the simulator's truth code.

What was asked for comes from the dataset's scenario.yaml: each perturbation's kind, size
(``magnitude_px``/``magnitude_mm``/``deg``/``factor``), direction or axis, pivot and schedule;
the nuisances; the content; the camera's timing. The geometry comes from setup.json: the
calibrated boxes, the overlap P (box A ∩ box B ∩ content rect), and the coarser projector's
pixel pitch at P's centroid. All of it is recomputed with the checker's own code
(``scripts/check_geometry.py``, ``scripts/check_timeline.py``), independent of ``sim/``. Per
frame, the recorded ground truth must then match:

  multiplier  every schedule re-evaluated here; continuous ones (drift, ramp, bump_then_hold,
              oscillate) may differ from it by half a quantum, the step of m moving the offset 0.02 px
  h_actual    each projector's calibrated homography after its requested moves, in the order
              listed: a shift by the recorded vector (|m| x size long, exactly across: the
              inner-edge normal, away from the partner; along: at right angles; or as asked), a
              rotation by m x the requested angle or a scale by 1 + m (s - 1) about the requested
              pivot, a keystone along the requested axis with k of the requested sign and k / m
              the same in every frame
  offset_mm   shifts: |sum of B's shift vectors - sum of A's|; rotation: 2 R sin(|theta| / 2),
              R = the overlap vertex farthest from the pivot; scale: |s - 1| R; keystone: the
              requested size at full strength. Every keystone and any mix of kinds is also
              measured by brute force over a dense grid of the overlap, on the verified h_actual
  offset_px   offset_mm / the pitch computed here
  h_rel       D_B D_A^-1 of the recorded h_actual, with D_p = h_actual,p h_cal,p^-1
  timeline    t_s, the nuisance state, the content shown, the camera and the visible markers

within 1e-6 mm (brute force: 1e-3 mm, as a grid slightly underestimates the maximum; matrices:
1e-9). setup.json must agree with the scenario on the source feed and the exposure. Stored PNGs
are re-hashed, ``--rerender K`` renders K spread frames and up to K stored ones again, and the
variants of a sweep that differ only in their perturbation must share every frame before the
earliest onset among them (``scripts/check_frames.py``). A check that could not run (no frame
hashes, fields a dataset predates) is listed under "skipped"; ``--strict`` fails on it.

Usage: python -m scripts.check_dataset out/shift_sweep [--rerender 10] [--strict]   (a dataset or a sweep)
Prints one JSON line per dataset and exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scripts.check_frames import check_paired, check_pngs, rerender
from scripts.check_geometry import Geometry, brute_force_mm, compose, relative_homography, transform
from scripts.check_timeline import Timeline, expected_multiplier, perturbation_specs
from scripts.dataset_files import read_lines, variants

TOL_MM = 1e-6
BRUTE_TOL_MM = 1e-3
QUANTUM_PX = 0.02
CONTINUOUS = ("drift", "ramp", "bump_then_hold", "oscillate")


@dataclass(frozen=True, eq=False)
class Request:
    """One perturbation as the scenario asks for it, with all that stays the same from frame to frame."""

    name: str  # the projector it moves
    kind: str
    size: float | None  # the offset asked for, mm (None: sized by deg or factor)
    full: float | None  # rotation angle (rad) or scale - 1 at full strength
    unit: float  # the offset it alone causes at full strength, mm
    pivot: np.ndarray  # rotation, scale, keystone
    reach: float  # the overlap vertex farthest from the pivot, mm
    vector: np.ndarray | None  # shift: the unit direction asked for (None: no inner edge); keystone: its axis
    quantum: float  # the step of m a continuous schedule moves in; 0 for exact schedules
    schedule: Any
    direction: Any  # as written: across, along or [x, y]


def _size_mm(spec: dict[str, Any], pitch: float) -> float | None:
    if "magnitude_px" in spec:
        return float(spec["magnitude_px"]) * pitch
    if "magnitude_mm" in spec:
        return float(spec["magnitude_mm"])
    return None


def _unit_vector(v: Any) -> np.ndarray:
    v = np.array(v, dtype=float)
    return v / math.hypot(*v)


def requests(scenario: dict[str, Any], geo: Geometry) -> list[Request]:
    """Every perturbation of the scenario, in the simulator's order, read with the checker's geometry."""
    out = []
    for name, spec in perturbation_specs(scenario):
        kind, size = spec["kind"], _size_mm(spec, geo.pitch)
        pivot = np.zeros(2) if kind == "shift" else geo.pivot(name, spec.get("pivot"))
        reach, full, unit, vector = geo.reach(pivot), None, abs(size or 0.0), None
        if kind == "rotation":
            full = (math.radians(float(spec["deg"])) if size is None
                    else math.copysign(2 * math.asin(min(1.0, abs(size) / (2 * reach))), size))
            unit = 2 * reach * abs(math.sin(full / 2))
        elif kind == "scale":
            full = float(spec["factor"]) - 1.0 if size is None else size / reach
            unit = abs(full) * reach
        elif kind == "keystone":
            axis = spec.get("axis", "x")
            vector = _unit_vector({"x": [1, 0], "y": [0, 1]}[axis] if isinstance(axis, str) else axis)
        else:
            direction = spec.get("direction", "across")
            if isinstance(direction, str):
                a = geo.across(name)
                vector = a if a is None or direction == "across" else np.array([-a[1], a[0]])
            else:
                vector = _unit_vector(direction)
        schedule = spec.get("schedule")
        continuous = isinstance(schedule, dict) and schedule.get("type") in CONTINUOUS
        quantum = QUANTUM_PX * geo.pitch / unit if continuous and unit > 0 else 0.0
        out.append(Request(name, kind, size, full, unit, pivot, reach, vector, quantum, schedule,
                           spec.get("direction", "across")))
    return out


def _check_shift(line: dict[str, Any], req: Request, v: np.ndarray, m: float, problems: list[str]) -> None:
    """A shift's recorded vector: |m| x the size asked for, pointing exactly the way asked for."""
    if abs(np.hypot(*v) - abs(m * req.size)) > TOL_MM:
        problems.append(f"frame {line['i']}: shift of {np.hypot(*v):.9f} mm, asked {abs(m * req.size):.9f} mm")
    if req.size == 0:
        return  # a zero-size shift has no direction to check
    if req.vector is None:
        problems.append(f"frame {line['i']}: projector {req.name} has no inner edge to shift {req.direction}")
        return
    cos = float(v @ req.vector) / np.hypot(*v) * math.copysign(1.0, m * req.size)
    if cos < 1 - 1e-9:
        problems.append(f"frame {line['i']}: shift of {req.name} is not {req.direction} (cos {cos:.9f})")


def _frame_expectation(line: dict[str, Any], t: Fraction, reqs: list[Request],
                       problems: list[str]) -> tuple[float | None, bool]:
    """(expected offset in mm, or None if only brute force can tell; whether to measure it by brute force)."""
    entries = line["perturbation"]
    if len(entries) != len(reqs):
        problems.append(f"frame {line['i']}: {len(entries)} perturbation tags for {len(reqs)} specs")
        return None, False
    active = []
    for req, entry in zip(reqs, entries, strict=True):
        m = expected_multiplier(req.schedule, t)
        if abs(entry["applied"] - m) > req.quantum / 2 + 1e-12 or entry["projector"] != req.name \
                or entry["kind"] != req.kind:
            problems.append(f"frame {line['i']}: {req.name} {req.kind} applied {entry['applied']}, expected {m}")
        if entry["applied"] != 0.0:  # within its quantum of the schedule: what the offset follows
            active.append((req, entry, entry["applied"]))
    if not active:
        return 0.0, False
    shifts = [(req, entry, m) for req, entry, m in active if req.kind == "shift"]
    for req, entry, m in shifts:  # checked whatever else is active
        _check_shift(line, req, np.array(entry["vector_mm"]), m, problems)
    if len(shifts) == len(active):
        total = {"a": np.zeros(2), "b": np.zeros(2)}
        for req, entry, _ in shifts:
            total[req.name] += np.array(entry["vector_mm"])
        return float(np.hypot(*(total["b"] - total["a"]))), False
    if len(active) == 1:
        req, entry, m = active[0]
        if "pivot_mm" in entry and not np.allclose(entry["pivot_mm"], req.pivot, atol=1e-9):
            problems.append(f"frame {line['i']}: pivot {entry['pivot_mm']} is not the requested one, {req.pivot.tolist()}")
        if req.kind == "rotation":
            return 2 * req.reach * abs(math.sin(m * req.full / 2)), False
        if req.kind == "scale":
            return abs(m * req.full) * req.reach, False
        if m == 1.0:
            return req.unit, True  # a keystone at full strength: the requested size, confirmed by brute force
    return None, True


def _check_homographies(line: dict[str, Any], reqs: list[Request], geo: Geometry, problems: list[str],
                        strengths: dict[int, float]) -> None:
    """h_actual from the request and the recorded multipliers; h_rel from h_actual.

    A keystone's recorded strength k is m x one fixed value, the one that gives the requested size
    at m = 1 (`strengths` keeps it per perturbation across frames), so every partial frame is tied
    to the full-strength one whose size is checked against the request.
    """
    moves: dict[str, list[np.ndarray]] = {"a": [], "b": []}
    for index, (req, entry) in enumerate(zip(reqs, line["perturbation"], strict=False)):
        m = entry["applied"]
        if m == 0.0:
            continue
        if req.kind == "shift":
            moves[req.name].append(transform("shift", 1.0, np.array(entry["vector_mm"]), np.zeros(2)))
        elif req.kind == "keystone":
            k = entry.get("k_per_mm", 0.0)
            if req.size and k and math.copysign(1.0, k) != math.copysign(1.0, m * req.size):
                problems.append(f"frame {line['i']}: keystone of {req.name} has k = {k}, of the wrong sign")
            unit = strengths.setdefault(index, k / m)
            if abs(k / m - unit) > 1e-9 * abs(unit):
                problems.append(f"frame {line['i']}: keystone of {req.name} has k / m = {k / m}, {unit} in other frames")
            moves[req.name].append(transform("keystone", k, req.vector, req.pivot))
        else:
            moves[req.name].append(transform(req.kind, m * req.full, np.zeros(2), req.pivot))
    h_actual = {n: np.array(line["truth"]["h_actual"][n]) for n in ("a", "b")}
    for n in ("a", "b"):
        if not np.allclose(h_actual[n], compose(geo.h[n], moves[n]), rtol=1e-12, atol=1e-9):
            problems.append(f"frame {line['i']}: h_actual of {n} is not its requested moves applied to h_cal")
    rel, got = relative_homography(h_actual, geo.h), np.array(line["truth"]["h_rel"])
    if np.abs(got - rel).max() > 1e-9 * max(1.0, np.abs(rel).max()):
        problems.append(f"frame {line['i']}: h_rel is not D_B D_A^-1 of the recorded h_actual")


def _setup_problems(setup: dict[str, Any], scenario: dict[str, Any], timeline: Timeline, n: int) -> list[str]:
    out = []
    available = bool((scenario.get("reference") or {}).get("available", False))
    if (setup.get("reference") or {}).get("available") != available:
        out.append(f"setup.json: reference.available is not {available}, as the scenario says")
    if setup["camera"].get("exposure_s") != float(timeline.exposure):
        out.append(f"setup.json: camera.exposure_s is not {float(timeline.exposure)}, as the scenario says")
    if n != timeline.n_frames:
        out.append(f"metadata.jsonl: {n} frames, the scenario makes {timeline.n_frames}")
    return out


def check(path: Path, k: int = 0) -> dict[str, Any]:
    setup = json.loads((path / "setup.json").read_text())
    scenario = yaml.safe_load((path / "scenario.yaml").read_text())
    lines = read_lines(path)
    if not lines:
        return {"dataset": str(path), "frames": 0, "skipped": [], "problems": ["metadata.jsonl: no frames"],
                "n_problems": 1, "ok": False}
    geo, timeline = Geometry(setup), Timeline(scenario, setup, lines[0])
    reqs = requests(scenario, geo)
    problems, skipped = _setup_problems(setup, scenario, timeline, len(lines)), set()
    worst, worst_px, worst_brute, n_closed, n_brute = 0.0, 0.0, 0.0, 0, 0
    brute_cache: dict[str, float] = {}
    strengths: dict[int, float] = {}
    for n, line in enumerate(lines):
        if line["i"] != n:
            problems.append(f"line {n}: frame index {line['i']}")
        expected, brute = _frame_expectation(line, timeline.phase + n * timeline.sample, reqs, problems)
        _check_homographies(line, reqs, geo, problems, strengths)
        timeline.check(line, problems, skipped)
        got = line["truth"]["offset_mm"]
        worst_px = max(worst_px, abs(line["truth"]["offset_px"] - got / geo.pitch))
        if expected is not None:
            worst = max(worst, abs(got - expected))
            n_closed += 1
        if brute:
            key = json.dumps(line["truth"]["h_actual"])
            if key not in brute_cache:
                brute_cache[key] = brute_force_mm(line, geo)
            worst_brute = max(worst_brute, abs(got - brute_cache[key]))
            n_brute += 1
    pngs, png_problems = check_pngs(path, lines)
    problems += png_problems
    rendered = rerender(path, lines, k) if k else {}
    problems += rendered.pop("problems", [])
    if "rerender_skipped" in rendered:
        skipped.add("rerender: " + rendered["rerender_skipped"])
    ok = (worst <= TOL_MM and worst_px <= TOL_MM / geo.pitch and worst_brute <= BRUTE_TOL_MM and not problems)
    return {"dataset": str(path), "frames": len(lines), "pitch_mm": geo.pitch, "closed_form_frames": n_closed,
            "max_offset_error_mm": worst, "max_offset_px_error": worst_px, "brute_force_frames": n_brute,
            "max_brute_force_error_mm": worst_brute, "pngs_checked": pngs, **rendered, "skipped": sorted(skipped),
            "problems": problems[:5], "n_problems": len(problems), "ok": ok}


def _non_negative(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be 0 (off) or more, got {value}")
    return value


def main(argv: list[str] | None = None) -> bool:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path, help="a dataset directory, or a sweep directory with variants.json")
    parser.add_argument("--rerender", type=_non_negative, default=0, metavar="K",
                        help="render K frames spread over each dataset, and up to K stored ones, again and compare")
    parser.add_argument("--strict", action="store_true", help="fail when a check had to be skipped")
    args = parser.parse_args(argv)
    names = variants(args.path)
    present = None if names is None else [v for v in names if (args.path / v).exists()]
    ok = True
    for path in [args.path] if present is None else [args.path / v for v in present]:
        result = check(path, args.rerender)
        ok &= result["ok"] and not (args.strict and result["skipped"])
        print(json.dumps(result))
    if present is not None:
        paired = check_paired(args.path, present)
        ok &= paired["ok"] and not (args.strict and paired["paired_skipped_no_hashes"])
        print(json.dumps(paired))
    print(json.dumps({"ok": ok}))
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
