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
              listed: a shift by the recorded vector (of length |m| x size, across = away from the
              partner, along = at right angles), a rotation by m x the requested angle or a scale
              by 1 + m (s - 1) about the requested pivot, a keystone along the requested axis about
              the pivot, with the recorded strength k of the requested sign and k / m the same in
              every frame
  offset_mm   shifts: |sum of B's shift vectors - sum of A's|; rotation: 2 R sin(|theta| / 2),
              R = the overlap vertex farthest from the pivot; scale: |s - 1| R; keystone: the
              requested size at full strength. Every keystone and any mix of kinds is also
              measured by brute force over a dense grid of the overlap, on the h_actual verified
              above, so a keystone's recorded strength is held to the size asked for
  offset_px   offset_mm / the pitch computed here
  h_rel       D_B D_A^-1 of the recorded h_actual, with D_p = h_actual,p h_cal,p^-1
  timeline    t_s, the nuisance state, the content shown, the camera and the visible markers
              (``scripts/check_timeline.py``)

within 1e-6 mm (brute force: 1e-3 mm, as a grid slightly underestimates the maximum; matrices:
1e-9). setup.json must agree with the scenario on the source feed and the exposure. Stored PNGs
are re-hashed, and ``--rerender K`` renders K spread frames and up to K stored ones again
(``scripts/check_frames.py``). For a
sweep directory, variants that differ only in their perturbation must have produced
bit-identical frames before the earliest onset among them: they share their seed.

A check that could not run (no frame hashes, fields a dataset predates) is listed under
"skipped" rather than passed silently; ``--strict`` makes any skip a failure.

Usage:
    python -m scripts.check_dataset out/shift_sweep [--rerender 10] [--strict]   (one dataset, or a sweep)

Prints one JSON line per dataset and exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
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


def _size_mm(spec: dict[str, Any], pitch: float) -> float | None:
    if "magnitude_px" in spec:
        return float(spec["magnitude_px"]) * pitch
    if "magnitude_mm" in spec:
        return float(spec["magnitude_mm"])
    return None


def _full_strength(name: str, spec: dict[str, Any], geo: Geometry) -> tuple[float | None, float]:
    """(rotation angle or scale - 1 at m = 1, else None; the offset it alone then causes, in mm)."""
    size = _size_mm(spec, geo.pitch)
    if spec["kind"] in ("shift", "keystone"):
        return None, abs(size)
    r = geo.reach(geo.pivot(name, spec.get("pivot")))
    if spec["kind"] == "rotation":
        full = (math.radians(float(spec["deg"])) if size is None
                else math.copysign(2 * math.asin(min(1.0, abs(size) / (2 * r))), size))
        return full, 2 * r * abs(math.sin(full / 2))
    full = float(spec["factor"]) - 1.0 if size is None else size / r
    return full, abs(full) * r


def _axis(spec: dict[str, Any]) -> np.ndarray:
    axis = spec.get("axis", "x")
    if isinstance(axis, str):
        return np.array([1.0, 0.0]) if axis == "x" else np.array([0.0, 1.0])
    v = np.array(axis, dtype=float)
    return v / math.hypot(*v)


def _check_shift(line: dict[str, Any], name: str, spec: dict[str, Any], v: np.ndarray, m: float, geo: Geometry,
                 problems: list[str]) -> None:
    """A shift's recorded vector: |m| x the requested size, pointing the requested way."""
    size = _size_mm(spec, geo.pitch)
    if abs(np.hypot(*v) - abs(m * size)) > TOL_MM:
        problems.append(f"frame {line['i']}: shift of {np.hypot(*v):.9f} mm, asked {abs(m * size):.9f} mm")
    if size == 0:
        return  # a zero-size shift has no direction to check
    partner = "a" if name == "b" else "b"
    away = geo.boxes[name].mean(axis=0) - geo.boxes[partner].mean(axis=0)
    cos = float(v @ away) / (np.hypot(*v) * np.hypot(*away)) * math.copysign(1.0, m * size)
    direction = spec.get("direction", "across")
    if (direction == "across" and cos < 0.9) or (direction == "along" and abs(cos) > 0.1):
        problems.append(f"frame {line['i']}: shift {direction} points {cos:+.3f} along the centre-to-centre line")
    if isinstance(direction, list):
        given = np.array(direction, dtype=float) * math.copysign(1.0, m * size)
        if float(v @ given) / (np.hypot(*v) * np.hypot(*given)) < 1 - 1e-9:
            problems.append(f"frame {line['i']}: shift does not follow the requested vector {direction}")


def _frame_expectation(line: dict[str, Any], t: Fraction, specs: list, geo: Geometry,
                       problems: list[str]) -> tuple[float | None, bool]:
    """(expected offset in mm, or None if only brute force can tell; whether to measure it by brute force)."""
    entries = line["perturbation"]
    if len(entries) != len(specs):
        problems.append(f"frame {line['i']}: {len(entries)} perturbation tags for {len(specs)} specs")
        return None, False
    active = []
    for (name, spec), entry in zip(specs, entries, strict=True):
        m = expected_multiplier(spec.get("schedule"), t)
        unit = _full_strength(name, spec, geo)[1]
        kind = spec["schedule"].get("type") if isinstance(spec.get("schedule"), dict) else None
        quantum = QUANTUM_PX * geo.pitch / unit if kind in CONTINUOUS and unit > 0 else 0.0
        if abs(entry["applied"] - m) > quantum / 2 + 1e-12 or entry["projector"] != name or entry["kind"] != spec["kind"]:
            problems.append(f"frame {line['i']}: {name} {spec['kind']} applied {entry['applied']}, expected {m}")
        m = entry["applied"]  # within its quantum of the schedule: what the offset follows
        if m != 0.0:
            active.append((name, spec, entry, m))
    if not active:
        return 0.0, False
    shifts = [(name, spec, entry, m) for name, spec, entry, m in active if spec["kind"] == "shift"]
    for name, spec, entry, m in shifts:  # checked whatever else is active
        _check_shift(line, name, spec, np.array(entry["vector_mm"]), m, geo, problems)
    if len(shifts) == len(active):
        total = {"a": np.zeros(2), "b": np.zeros(2)}
        for name, _, entry, _ in shifts:
            total[name] += np.array(entry["vector_mm"])
        return float(np.hypot(*(total["b"] - total["a"]))), False
    if len(active) == 1:
        name, spec, entry, m = active[0]
        pivot = geo.pivot(name, spec.get("pivot"))
        if "pivot_mm" in entry and not np.allclose(entry["pivot_mm"], pivot, atol=1e-9):
            problems.append(f"frame {line['i']}: pivot {entry['pivot_mm']} is not the requested {spec.get('pivot', 'centre')}")
        full, unit = _full_strength(name, spec, geo)
        if spec["kind"] == "rotation":
            return 2 * geo.reach(pivot) * abs(math.sin(m * full / 2)), False
        if spec["kind"] == "scale":
            return abs(m * full) * geo.reach(pivot), False
        if m == 1.0:
            return unit, True  # a keystone at full strength: the requested size, confirmed by brute force
    return None, True


def _check_homographies(line: dict[str, Any], specs: list, geo: Geometry, problems: list[str],
                        strengths: dict[int, float]) -> None:
    """h_actual from the request and the recorded multipliers; h_rel from h_actual.

    A keystone's recorded strength k is m x one fixed value, the one that gives the requested size
    at m = 1 (`strengths` keeps it per perturbation across frames), so every partial frame is tied
    to the full-strength one whose size is checked against the request.
    """
    moves: dict[str, list[np.ndarray]] = {"a": [], "b": []}
    for index, ((name, spec), entry) in enumerate(zip(specs, line["perturbation"], strict=False)):
        m = entry["applied"]
        if m == 0.0:
            continue
        if spec["kind"] == "shift":
            moves[name].append(transform("shift", 1.0, np.array(entry["vector_mm"]), np.zeros(2)))
            continue
        pivot = geo.pivot(name, spec.get("pivot"))
        if spec["kind"] == "keystone":
            k, size = entry.get("k_per_mm", 0.0), _size_mm(spec, geo.pitch)
            if size and k and math.copysign(1.0, k) != math.copysign(1.0, m * size):
                problems.append(f"frame {line['i']}: keystone of {name} has k = {k}, of the wrong sign")
            unit = strengths.setdefault(index, k / m)
            if abs(k / m - unit) > 1e-9 * abs(unit):
                problems.append(f"frame {line['i']}: keystone of {name} has k / m = {k / m}, {unit} in other frames")
            moves[name].append(transform("keystone", k, _axis(spec), pivot))
        else:
            moves[name].append(transform(spec["kind"], m * _full_strength(name, spec, geo)[0], np.zeros(2), pivot))
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
    geo, timeline, specs = Geometry(setup), Timeline(scenario, setup, lines[0]), perturbation_specs(scenario)
    problems, skipped = _setup_problems(setup, scenario, timeline, len(lines)), set()
    worst, worst_px, worst_brute, n_closed, n_brute = 0.0, 0.0, 0.0, 0, 0
    brute_cache: dict[str, float] = {}
    strengths: dict[int, float] = {}
    for n, line in enumerate(lines):
        if line["i"] != n:
            problems.append(f"line {n}: frame index {line['i']}")
        expected, brute = _frame_expectation(line, timeline.phase + n * timeline.sample, specs, geo, problems)
        _check_homographies(line, specs, geo, problems, strengths)
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


def main(argv: list[str] | None = None) -> bool:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path, help="a dataset directory, or a sweep directory with variants.json")
    parser.add_argument("--rerender", type=int, default=0, metavar="K",
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
