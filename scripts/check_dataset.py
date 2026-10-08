"""Check a generated dataset against what the scenario asked for, independently of the simulator's truth code.

What was asked for comes from the dataset's scenario.yaml: each perturbation's kind, size
(``magnitude_px``/``magnitude_mm``/``deg``/``factor``), direction, pivot and schedule. The
geometry comes from setup.json: the calibrated boxes, the overlap P (box A ∩ box B ∩ content
rect), and the coarser projector's pixel pitch at P's centroid. All of it is recomputed here with
this file's own geometry code. Per frame, the recorded ground truth must then match:

  multiplier  step, staircase and none schedules re-evaluated here; continuous ones within half
              a 0.02 px quantum of the schedule
  shift       offset = |sum of B's shift vectors - sum of A's|, each of length
              |m| x size (mm), across = pointing away from the partner, along = at right
              angles to it; the moved projector's h_actual = T(shift) h_cal, the other's unchanged
  rotation    offset = 2 R sin(|theta| / 2), R = the overlap vertex farthest from the pivot
  scale       offset = |s - 1| R
  keystone    offset = the requested size at full strength; otherwise, and for any mix of
              kinds, a brute-force maximum over a dense grid of the overlap
  offset_px   offset_mm / the pitch computed here

within 1e-6 mm (brute force: 1e-3 mm, as a grid slightly underestimates the maximum). For a
sweep directory, variants that differ only in their perturbation must have produced
bit-identical frames before the earliest onset among them: they share their seed.

Usage: python -m scripts.check_dataset out/shift_sweep   (one dataset, or a sweep directory)
Prints one JSON line per dataset and exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import yaml

TOL_MM = 1e-6
BRUTE_TOL_MM = 1e-3
QUANTUM_PX = 0.02


# -- this file's own plane geometry ------------------------------------------------------------
def _apply(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.atleast_2d(pts)
    w = h[2, 0] * pts[:, 0] + h[2, 1] * pts[:, 1] + h[2, 2]
    return np.stack([(h[0, 0] * pts[:, 0] + h[0, 1] * pts[:, 1] + h[0, 2]) / w,
                     (h[1, 0] * pts[:, 0] + h[1, 1] * pts[:, 1] + h[1, 2]) / w], axis=-1)


def _box(h: np.ndarray, resolution: list[int]) -> np.ndarray:
    w, hh = resolution
    return _apply(h, np.array([[-0.5, -0.5], [w - 0.5, -0.5], [w - 0.5, hh - 0.5], [-0.5, hh - 0.5]]))


def _ccw(poly: np.ndarray) -> np.ndarray:
    area = np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
    return poly if area > 0 else poly[::-1]


def _clip(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman: the part of convex `subject` inside convex `clip`."""
    out = list(_ccw(subject))
    clip = _ccw(clip)
    for a, b in zip(clip, np.roll(clip, -1, axis=0)):
        inside = [(b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0 for p in out]
        new = []
        for i, p in enumerate(out):
            q, q_in = out[i - 1], inside[i - 1]
            if inside[i] != q_in:
                d = p - q
                e = b - a
                t = ((a[0] - q[0]) * e[1] - (a[1] - q[1]) * e[0]) / (d[0] * e[1] - d[1] * e[0])
                new.append(q + t * d)
            if inside[i]:
                new.append(p)
        out = new
    return np.array(out)


def _centroid(poly: np.ndarray) -> np.ndarray:
    x, y = poly[:, 0], poly[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    return np.array([np.sum((x + np.roll(x, -1)) * cross), np.sum((y + np.roll(y, -1)) * cross)]) / (3 * np.sum(cross))


def _pitch(h: np.ndarray, at_mm: np.ndarray) -> float:
    u, v = _apply(np.linalg.inv(h), at_mm)[0]
    w = h[2, 0] * u + h[2, 1] * v + h[2, 2]
    return math.sqrt(abs(np.linalg.det(h) / w**3))


class Geometry:
    def __init__(self, setup: dict[str, Any]) -> None:
        self.h = {n: np.array(p["h_cal_px_to_mm"]) for n, p in setup["projectors"].items()}
        self.res = {n: p["resolution"] for n, p in setup["projectors"].items()}
        self.boxes = {n: _box(self.h[n], self.res[n]) for n in self.h}
        x0, y0, x1, y1 = setup["content_rect_mm"]
        rect = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)
        self.overlap = _clip(_clip(self.boxes["a"], self.boxes["b"]), rect)
        self.pitch = max(_pitch(self.h[n], _centroid(self.overlap)) for n in self.h)

    def pivot(self, name: str, spec: Any) -> np.ndarray:
        if spec in (None, "centre"):
            w, h = self.res[name]
            return _apply(self.h[name], np.array([(w - 1) / 2, (h - 1) / 2]))[0]
        if spec == "overlap_centre":
            return _centroid(self.overlap)
        if spec == "far_corner":
            d = np.hypot(*(self.boxes[name] - _centroid(self.overlap)).T)
            return self.boxes[name][int(np.nonzero(d >= d.max() - 1e-9)[0][0])]
        return np.array(spec, dtype=float)

    def reach(self, pivot: np.ndarray) -> float:
        return float(np.hypot(*(self.overlap - pivot).T).max())


# -- what the scenario asked for ---------------------------------------------------------------
def _specs(scenario: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    out = []
    perturbation = scenario.get("perturbation")
    if isinstance(perturbation, dict):
        for name in ("a", "b"):  # the simulator's order: projectors by name, then list order
            items = perturbation.get(name)
            if items is not None:
                out += [(name, s) for s in ([items] if isinstance(items, dict) else items)]
    return out


def _size_mm(spec: dict[str, Any], pitch: float) -> float | None:
    if "magnitude_px" in spec:
        return float(spec["magnitude_px"]) * pitch
    if "magnitude_mm" in spec:
        return float(spec["magnitude_mm"])
    return None


def expected_multiplier(schedule: Any, t: Fraction) -> float | None:
    """None, step and staircase schedules, re-evaluated here; None for the continuous ones."""
    if schedule in (None, "none") or (isinstance(schedule, dict) and schedule.get("type") == "none"):
        return 0.0
    t0 = Fraction(str(schedule.get("t0_s", 0)))
    if schedule["type"] == "step":
        return 1.0 if t >= t0 else 0.0
    if schedule["type"] == "staircase":
        if t < t0:
            return 0.0
        k = int((t - t0) // Fraction(str(schedule["hold_s"])))
        return float(schedule["levels"][min(k, len(schedule["levels"]) - 1)])
    return None


def brute_force_mm(line: dict[str, Any], geo: Geometry) -> float:
    d = {n: np.array(line["truth"]["h_actual"][n]) @ np.linalg.inv(geo.h[n]) for n in ("a", "b")}
    poly = geo.overlap
    lo, hi = poly.min(axis=0), poly.max(axis=0)
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 401), np.linspace(lo[1], hi[1], 401))
    pts = np.stack([gx.ravel(), gy.ravel()], axis=-1)
    inside = np.ones(len(pts), bool)
    for a, b in zip(_ccw(poly), np.roll(_ccw(poly), -1, axis=0)):
        inside &= (b[0] - a[0]) * (pts[:, 1] - a[1]) - (b[1] - a[1]) * (pts[:, 0] - a[0]) >= -1e-9
    edge_pts = [p + np.linspace(0, 1, 2001)[:, None] * (q - p) for p, q in zip(poly, np.roll(poly, -1, axis=0))]
    pts = np.vstack([pts[inside], poly, *edge_pts])
    return float(np.hypot(*(_apply(d["b"], pts) - _apply(d["a"], pts)).T).max())


def _frame_expectation(line: dict[str, Any], specs: list, geo: Geometry, problems: list[str]) -> tuple[float | None, bool]:
    """(expected offset in mm, or None if only brute force can tell; whether brute force is needed)."""
    entries = line["perturbation"]
    if len(entries) != len(specs):
        problems.append(f"frame {line['i']}: {len(entries)} perturbation tags for {len(specs)} specs")
        return None, False
    t = Fraction(repr(line["t_s"]))
    active = []
    for (name, spec), entry in zip(specs, entries):
        m = expected_multiplier(spec.get("schedule"), t)
        if m is None:  # continuous: quantized from the schedule's own value
            size = _size_mm(spec, geo.pitch)
            if size:
                quantum = QUANTUM_PX * geo.pitch / abs(size)
                if abs(entry["applied"] - entry["scheduled"]) > quantum / 2 + 1e-12:
                    problems.append(f"frame {line['i']}: applied {entry['applied']} is off the schedule's "
                                    f"{entry['scheduled']}")
            m = entry["applied"]
        elif entry["applied"] != m or entry["projector"] != name or entry["kind"] != spec["kind"]:
            problems.append(f"frame {line['i']}: {name} {spec['kind']} applied {entry['applied']}, expected {m}")
        if m != 0.0:
            active.append((name, spec, entry, m))
    if not active:
        return 0.0, False
    if all(spec["kind"] == "shift" for _, spec, _, _ in active):
        total = {"a": np.zeros(2), "b": np.zeros(2)}
        for name, spec, entry, m in active:
            v = np.array(entry["vector_mm"])
            size = _size_mm(spec, geo.pitch)
            if abs(np.hypot(*v) - abs(m * size)) > TOL_MM:
                problems.append(f"frame {line['i']}: shift of {np.hypot(*v):.9f} mm, asked {abs(m * size):.9f} mm")
            total[name] += v
            if size == 0:
                continue  # a zero-size shift has no direction to check
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
        for name in ("a", "b"):
            h_act = np.array(line["truth"]["h_actual"][name])
            want = geo.h[name] if not total[name].any() else np.array([[1, 0, total[name][0]], [0, 1, total[name][1]], [0, 0, 1.0]]) @ geo.h[name]
            if not np.allclose(h_act, want, rtol=1e-12, atol=1e-9):
                problems.append(f"frame {line['i']}: h_actual of {name} is not its shift of h_cal")
        return float(np.hypot(*(total["b"] - total["a"]))), False
    if len(active) == 1:
        name, spec, entry, m = active[0]
        pivot = geo.pivot(name, spec.get("pivot"))
        if "pivot_mm" in entry and not np.allclose(entry["pivot_mm"], pivot, atol=1e-9):
            problems.append(f"frame {line['i']}: pivot {entry['pivot_mm']} is not the requested {spec.get('pivot', 'centre')}")
        r = geo.reach(pivot)
        size = _size_mm(spec, geo.pitch)
        if spec["kind"] == "rotation":
            full = math.radians(float(spec["deg"])) if size is None else math.copysign(2 * math.asin(abs(size) / (2 * r)), size)
            return 2 * r * abs(math.sin(m * full / 2)), False
        if spec["kind"] == "scale":
            full = float(spec["factor"]) - 1.0 if size is None else size / r
            return abs(m * full) * r, False
        if spec["kind"] == "keystone" and m == 1.0:
            return abs(size), False
    return None, True


def check(path: Path) -> dict[str, Any]:
    setup = json.loads((path / "setup.json").read_text())
    scenario = yaml.safe_load((path / "scenario.yaml").read_text())
    lines = [json.loads(s) for s in (path / "metadata.jsonl").read_text().splitlines()]
    geo = Geometry(setup)
    specs = _specs(scenario)
    problems: list[str] = []
    worst, worst_px, worst_brute, n_closed, n_brute = 0.0, 0.0, 0.0, 0, 0
    for line in lines:
        expected, brute = _frame_expectation(line, specs, geo, problems)
        got = line["truth"]["offset_mm"]
        worst_px = max(worst_px, abs(line["truth"]["offset_px"] - got / geo.pitch))
        if brute:
            worst_brute = max(worst_brute, abs(got - brute_force_mm(line, geo)))
            n_brute += 1
        elif expected is not None:
            worst = max(worst, abs(got - expected))
            n_closed += 1
    ok = (worst <= TOL_MM and worst_px <= TOL_MM / geo.pitch and worst_brute <= BRUTE_TOL_MM and not problems)
    return {"dataset": str(path), "frames": len(lines), "pitch_mm": geo.pitch, "closed_form_frames": n_closed,
            "max_offset_error_mm": worst, "max_offset_px_error": worst_px, "brute_force_frames": n_brute,
            "max_brute_force_error_mm": worst_brute, "problems": problems[:5], "n_problems": len(problems), "ok": ok}


def check_paired(root: Path, variants: list[str]) -> dict[str, Any]:
    """Variants differing only in their perturbation must share every frame before the first onset."""
    groups: dict[str, list[str]] = {}
    for v in variants:
        scenario = yaml.safe_load((root / v / "scenario.yaml").read_text())
        rest = {k: val for k, val in copy.deepcopy(scenario).items() if k != "perturbation"}
        groups.setdefault(json.dumps(rest, sort_keys=True, default=str), []).append(v)
    checked, same = 0, True
    for members in groups.values():
        hashes, onsets = {}, []
        for v in members:
            lines = [json.loads(s) for s in (root / v / "metadata.jsonl").read_text().splitlines()]
            hashes[v] = [line["frame_sha256"] for line in lines]
            moved = [line["i"] for line in lines if any(e["applied"] != 0.0 for e in line["perturbation"])]
            if moved:
                onsets.append(moved[0])
        if len(members) < 2 or any(h is None for hs in hashes.values() for h in hs):
            continue
        first = min(onsets) if onsets else min(len(h) for h in hashes.values())
        reference = hashes[members[0]][:first]
        same &= all(hashes[v][:first] == reference for v in members)
        checked += first * len(members)
    return {"paired_groups": len(groups), "paired_frames_checked": checked, "paired_identical": same, "ok": same}


def main(argv: list[str] | None = None) -> bool:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    index = args.path / "variants.json"
    ok = True
    if index.exists():
        variants = [v["dir"] for v in json.loads(index.read_text())["variants"] if (args.path / v["dir"]).exists()]
        for v in variants:
            result = check(args.path / v)
            ok &= result["ok"]
            print(json.dumps(result))
        paired = check_paired(args.path, variants)
        ok &= paired["ok"]
        print(json.dumps(paired))
    else:
        result = check(args.path)
        ok &= result["ok"]
        print(json.dumps(result))
    print(json.dumps({"ok": ok}))
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
