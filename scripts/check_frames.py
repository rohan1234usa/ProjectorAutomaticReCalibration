"""A dataset's frames: do the stored PNGs and freshly rendered frames match the recorded hashes?

A dataset records the sha256 of every frame it rendered (``--frames sample`` or ``all``) and
stores some of them as 16-bit PNGs. Two checks follow from that:

  stored     every PNG is read back and hashed, so a PNG that was overwritten, truncated or
             written from another frame is caught;
  rerender   frames are rendered again from the dataset's own scenario.yaml and compared with
             what was recorded, hash and metadata line. This is how a stored dataset proves that
             the current code still makes the frames it describes: the "no frame changed" check
             after every simulator change.

Re-rendering necessarily runs the simulator, so this is the one place a checker uses sim/. It
imports it inside :func:`rerender`, so the rest of the checker stays independent of it; and
what it compares against (hashes and metadata lines) the rest of check_dataset verifies on its own.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from scripts.dataset_files import pixel_hash, read_png


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


def _stored_mismatches(stored: Any, fresh: Any, prefix: str, out: list[str], added: set[str]) -> None:
    """Fields of the stored line that the fresh one lacks or disagrees with; fields only the fresh one has go to `added`."""
    if isinstance(stored, dict) and isinstance(fresh, dict):
        for key in fresh.keys() - stored.keys():
            added.add(f"{prefix}{key}")
        for key, value in stored.items():
            if key not in fresh:
                out.append(f"{prefix}{key}")
            else:
                _stored_mismatches(value, fresh[key], f"{prefix}{key}.", out, added)
    elif stored != fresh:
        out.append(prefix.rstrip("."))


def frames_to_render(lines: list[dict[str, Any]], k: int) -> list[int]:
    """k frames spread evenly over the run, plus every frame with a stored PNG."""
    n = len(lines)
    spread = {round(j * (n - 1) / max(1, k - 1)) for j in range(k)} if n else set()
    return sorted(spread | {line["i"] for line in lines if line.get("png")})


def rerender(path: Path, lines: list[dict[str, Any]], k: int) -> dict[str, Any]:
    """Render k spread frames (and every PNG frame) again and compare hashes and metadata lines.

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
        fields: list[str] = []
        _stored_mismatches(stored, json.loads(json.dumps(source.truth(i, state))), "", fields, added)
        if fields:
            problems.append(f"frame {i}: re-rendered metadata differs in {fields}")
    return {"rerendered": len(chosen), "fields_added_since": sorted(added), "problems": problems}
