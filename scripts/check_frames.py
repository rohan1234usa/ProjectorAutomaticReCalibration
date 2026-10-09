"""A dataset's frames: do the stored PNGs and freshly rendered frames match the recorded hashes?

A dataset records the sha256 of every frame it rendered (``--frames sample`` or ``all``) and
stores some of them as 16-bit PNGs. Three checks follow from that:

  stored     every PNG is read back and hashed, so a PNG that was overwritten, truncated or
             written from another frame is caught;
  rerender   frames are rendered again from the dataset's own scenario.yaml and compared with
             what was recorded, hash and metadata line. This is how a stored dataset proves that
             the current code still makes the frames it describes: the "no frame changed" check
             after every simulator change;
  paired     sweep variants that differ only in their perturbation share their seed, so every
             frame before the earliest onset among them must have the same hash in all of them.

Re-rendering necessarily runs the simulator, so this is the one place a checker uses sim/. It
imports it inside :func:`rerender`, so the rest of the checker stays independent of it; and
what it compares against (hashes and metadata lines) the rest of check_dataset verifies on its own.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from scripts.dataset_files import diff_paths, pixel_hash, present, read_lines, read_png


def check_pngs(path: Path, lines: list[dict[str, Any]]) -> tuple[int, list[str]]:
    """(PNGs checked, problems): every stored PNG against its frame's recorded hash."""
    problems, n = [], 0
    for line in lines:
        if not line.get("png"):
            continue
        n += 1
        try:
            frame = read_png(path / line["png"])
        except FileNotFoundError:
            problems.append(f"frame {line['i']}: {line['png']} is missing or unreadable")
            continue
        if pixel_hash(frame) != line["frame_sha256"]:
            problems.append(f"frame {line['i']}: {line['png']} does not match the frame's recorded hash")
    return n, problems


def _spread(items: list[int], k: int) -> set[int]:
    """At most k of `items`, evenly spread: all of them when there are no more than k."""
    if len(items) <= k:
        return set(items)
    return {items[round(j * (len(items) - 1) / max(1, k - 1))] for j in range(k)}


def frames_to_render(lines: list[dict[str, Any]], k: int) -> list[int]:
    """k frames spread evenly over the run, plus up to k of those with a stored PNG (spread too).

    A `--frames sample` dataset stores a handful of PNGs, at regular frames and where the geometry
    changes, and all of them are rendered again; one written with `--frames all` stores every
    frame, which would turn a spot check into a full re-render.
    """
    return sorted(_spread(list(range(len(lines))), k) | _spread([line["i"] for line in lines if line.get("png")], k))


def rerender(path: Path, lines: list[dict[str, Any]], k: int) -> dict[str, Any]:
    """Render k spread frames (and up to k stored ones) again and compare hashes and metadata lines.

    Returns how many were rendered, the metadata fields the current code writes that the dataset
    predates (information: the timeline check reports what it could not check), and the problems.
    """
    if any(line["frame_sha256"] is None for line in lines):
        return {"rerendered": 0, "rerender_skipped": "no frame hashes (--frames none)"}
    from sim.frames import FrameSource  # the one place a checker runs the simulator (see the docstring)
    from sim.scenario import scenario_from_dict

    variant = json.loads((path / "dataset.json").read_text()).get("variant")
    source = FrameSource(scenario_from_dict(yaml.safe_load((path / "scenario.yaml").read_text()), variant=variant))
    problems: list[str] = []
    added: set[str] = set()
    chosen = frames_to_render(lines, k)
    for i in chosen:
        line = lines[i]
        state = source.state(i)
        if pixel_hash(source.frame(i, state)) != line["frame_sha256"]:
            problems.append(f"frame {i}: the re-rendered frame does not match the recorded hash")
        stored = {key: value for key, value in line.items() if key not in ("frame_sha256", "png")}
        differ = diff_paths(stored, json.loads(json.dumps(source.truth(i, state))))
        added |= {path for path in differ if not present(stored, path)}  # written by newer code
        fields = [path for path in differ if present(stored, path)]
        if fields:
            problems.append(f"frame {i}: re-rendered metadata differs in {fields}")
    return {"rerendered": len(chosen), "fields_added_since": sorted(added), "problems": problems}


def check_paired(root: Path, names: list[str]) -> dict[str, Any]:
    """Variants differing only in their perturbation must share every frame before the first onset."""
    groups: dict[str, list[str]] = {}
    for v in names:
        scenario = yaml.safe_load((root / v / "scenario.yaml").read_text())
        rest = {key: value for key, value in scenario.items() if key != "perturbation"}
        groups.setdefault(json.dumps(rest, sort_keys=True, default=str), []).append(v)
    checked, same, singletons, unhashed = 0, True, [], []
    for members in groups.values():
        if len(members) < 2:
            singletons += members
            continue
        hashes, onsets = {}, []
        for v in members:
            lines = read_lines(root / v)
            hashes[v] = [line["frame_sha256"] for line in lines]
            moved = [line["i"] for line in lines if any(e["applied"] != 0.0 for e in line["perturbation"])]
            if moved:
                onsets.append(moved[0])
        if any(h is None for hs in hashes.values() for h in hs):
            unhashed.append(members)
            continue
        first = min(onsets) if onsets else min(len(h) for h in hashes.values())
        reference = hashes[members[0]][:first]
        same &= all(hashes[v][:first] == reference for v in members)
        checked += first * len(members)
    return {"paired_groups": len(groups), "paired_singletons": singletons, "paired_skipped_no_hashes": unhashed,
            "paired_frames_checked": checked, "paired_identical": same, "ok": same}
