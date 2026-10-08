"""Datasets: one scenario variant written to disk with its ground truth, reproducibly.

A dataset directory holds:

  scenario.yaml    the resolved variant (no extends, no sweep): enough to re-render every frame
  setup.json       exactly what the detector may read: the blending setup, the marker layout,
                   the locked camera settings. Never any ground truth.
  metadata.jsonl   one line per frame: time, content, ground truth, perturbation and nuisance
                   state, and the sha256 of the frame's 16-bit pixels. Only the evaluation
                   harness reads it.
  frames/          optional 16-bit PNGs (about 8 MB each at 3840 x 1600)
  dataset.json     variant, scenario hash, library versions and platform
  timing.json      how long it took (not reproducible, so kept apart)

Frames are written only on request, because any frame can be re-rendered from scenario.yaml
(``sim/frames.py``), and the stored hash proves the re-rendered frame is the one described.
Determinism: metadata is written with sorted keys and Python's shortest exact float repr, so two
runs give byte-identical files. Pixel hashes depend on the machine and OpenCV build, which is
why dataset.json records them.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from sim.frames import FrameSource
from sim.scenario import Scenario, scenario_from_dict

FRAME_MODES = ("none", "sample", "all")


def frame_hash(frame: np.ndarray) -> str:
    """sha256 of the frame's pixels as little-endian 16-bit numbers, with its shape."""
    digest = hashlib.sha256(str(frame.shape).encode())
    digest.update(np.ascontiguousarray(frame, dtype="<u2").tobytes())
    return digest.hexdigest()


def dumps(obj: Any, indent: int | None = None) -> str:
    """Deterministic JSON: sorted keys, no NaN, compact unless indented."""
    separators = (",", ":") if indent is None else (",", ": ")
    return json.dumps(obj, sort_keys=True, separators=separators, allow_nan=False, indent=indent)


def environment() -> dict[str, Any]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                                cwd=Path(__file__).resolve().parent).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "git_commit": commit,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "pyyaml": yaml.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def write_dataset(
    scenario: Scenario,
    out_dir: str | Path,
    frames: str = "none",
    every: int = 600,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Write one variant. frames: none (truth only, no rendering), sample, or all.

    ``sample`` renders and hashes every frame but stores a PNG only every `every` frames and at
    each change of a projector's geometry (a perturbation's onset or step).
    """
    if frames not in FRAME_MODES:
        raise ValueError(f"frames must be one of {FRAME_MODES}, got {frames!r}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    source = FrameSource(scenario)
    scenario_text = yaml.safe_dump(scenario.data, sort_keys=False, allow_unicode=True)
    (out / "scenario.yaml").write_text(scenario_text)
    (out / "setup.json").write_text(dumps(source.setup_dict(), indent=1) + "\n")
    if frames != "none":
        (out / "frames").mkdir(exist_ok=True)
    render_s, written, previous = 0.0, 0, None
    with open(out / "metadata.jsonl", "w") as fh:
        for i in range(len(source)):
            state = source.state(i)
            line = source.truth(i, state)
            line["frame_sha256"], line["png"] = None, None
            if frames != "none":
                t0 = time.perf_counter()
                frame = source.frame(i, state)
                render_s += time.perf_counter() - t0
                line["frame_sha256"] = frame_hash(frame)
                moved = previous is None or state.applied != previous.applied  # the geometry just changed
                if frames == "all" or i % every == 0 or moved:
                    line["png"] = f"frames/{i:06d}.png"
                    cv2.imwrite(str(out / line["png"]), frame if frame.ndim == 2 else frame[..., ::-1])
                    written += 1
            fh.write(dumps(line) + "\n")
            previous = state
            if progress:
                progress(i + 1, len(source))
    total_s = time.perf_counter() - started
    info = {
        "scenario": scenario.name,
        "variant": scenario.variant,
        "scenario_sha256": hashlib.sha256(scenario_text.encode()).hexdigest(),
        "n_frames": len(source),
        "frames": frames,
        "pngs": written,
        "environment": environment(),
    }
    (out / "dataset.json").write_text(dumps(info, indent=1) + "\n")
    timing = {"total_s": round(total_s, 3), "render_s": round(render_s, 3),
              "render_ms_per_frame": round(1000 * render_s / max(1, len(source)), 2) if frames != "none" else None}
    (out / "timing.json").write_text(dumps(timing, indent=1) + "\n")
    return {"dir": str(out), **info, **timing}


@dataclass(eq=False)
class Dataset:
    """A dataset on disk. Frames come from the PNGs when present, else are re-rendered and checked."""

    path: Path
    setup: dict[str, Any]
    metadata: list[dict[str, Any]]
    info: dict[str, Any]
    _source: FrameSource | None = field(default=None, repr=False)

    def scenario(self) -> Scenario:
        data = yaml.safe_load((self.path / "scenario.yaml").read_text())
        return scenario_from_dict(data, variant=self.info.get("variant"))

    def source(self) -> FrameSource:
        if self._source is None:
            self._source = FrameSource(self.scenario())
        return self._source

    def frame(self, i: int) -> np.ndarray:
        line = self.metadata[i]
        if line.get("png"):
            frame = cv2.imread(str(self.path / line["png"]), cv2.IMREAD_UNCHANGED)
            frame = frame if frame.ndim == 2 else frame[..., ::-1]
        else:
            frame = self.source().frame(i)
        if line.get("frame_sha256") and frame_hash(frame) != line["frame_sha256"]:
            raise ValueError(f"{self.path}: frame {i} does not match its recorded hash")
        return frame


def read_dataset(path: str | Path) -> Dataset:
    path = Path(path)
    setup = json.loads((path / "setup.json").read_text())
    info = json.loads((path / "dataset.json").read_text())
    with open(path / "metadata.jsonl") as fh:
        metadata = [json.loads(line) for line in fh]
    return Dataset(path, setup, metadata, info)
