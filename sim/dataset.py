"""Datasets: one scenario variant written to disk with its ground truth, reproducibly.

A dataset directory holds:

  scenario.yaml    the resolved variant (no extends, no sweep): enough to re-render every frame
  setup.json       exactly what the detector may read: the blending setup, the marker layout,
                   the locked camera settings. Never any ground truth.
  metadata.jsonl   one line per frame: time, content, ground truth, perturbation and nuisance
                   state, and the sha256 of the frame's 16-bit pixels. Only the evaluation
                   harness reads it.
  frames/          optional 16-bit PNGs (about 8 MB each at 3840 x 1600)
  dataset.json     variant, scenario hash, the code's commit (and whether sim/ had uncommitted
                   changes), library versions and platform
  timing.json      how long it took (not reproducible, so kept apart)

Frames are written only on request, because any frame can be re-rendered from scenario.yaml
(``sim/frames.py``). With ``--frames sample`` or ``all`` the stored hash proves the re-rendered
frame is the one described; ``--frames none`` stores no hash, so nothing can be checked.
Determinism: metadata is written with sorted keys and Python's shortest exact float repr, so two
runs give byte-identical files. Pixel hashes depend on the machine and OpenCV build, which is
why dataset.json records them.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
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


def _git(*args: str) -> str | None:
    """A git command's output, run in sim/ (so pathspec "." means the simulator); None outside a checkout."""
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True,
                              cwd=Path(__file__).resolve().parent).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def environment() -> dict[str, Any]:
    """What made the dataset: the commit, whether sim/ differed from it, library versions, platform."""
    commit, changes = _git("rev-parse", "HEAD"), _git("status", "--porcelain", "--", ".")
    return {
        "git_commit": None if commit is None else commit.strip(),
        "git_dirty": None if changes is None else bool(changes.strip()),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "pyyaml": yaml.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def _write_range(scenario: Scenario, out: Path, frames: str, every: int, lo: int, hi: int, path: Path,
                 progress: Callable[[int, int], None] | None = None) -> tuple[float, int]:
    """Metadata lines (and PNGs) for frames lo..hi-1 into `path`; returns (render seconds, PNGs written)."""
    source = FrameSource(scenario)
    render_s, written = 0.0, 0
    previous = source.state(lo - 1) if lo > 0 else None  # so a range starts its PNG choices like a full run
    with open(path, "w") as fh:
        for i in range(lo, hi):
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
                progress(i + 1, hi)
    return render_s, written


def _write_part(data: dict[str, Any], variant: str, out: str, frames: str, every: int, lo: int, hi: int,
                part: str) -> tuple[float, int]:
    """One worker's share of a variant's frames (runs in a separate process)."""
    scenario = scenario_from_dict(data, variant=variant)
    return _write_range(scenario, Path(out), frames, every, lo, hi, Path(part))


def _init_worker(threads: int) -> None:
    cv2.setNumThreads(threads)


def process_pool(jobs: int) -> ProcessPoolExecutor:
    """`jobs` fresh worker processes ("spawn": nothing inherited), sharing the CPU's threads between them."""
    threads = max(1, (os.cpu_count() or 1) // jobs)
    context = multiprocessing.get_context("spawn")
    return ProcessPoolExecutor(jobs, mp_context=context, initializer=_init_worker, initargs=(threads,))


def write_variant(data: dict[str, Any], variant: str, out_dir: str, frames: str, every: int, jobs: int = 1) -> dict[str, Any]:
    """:func:`write_dataset` for a variant given as its resolved data (picklable, for worker processes)."""
    return write_dataset(scenario_from_dict(data, variant=variant), out_dir, frames=frames, every=every, jobs=jobs)


def write_dataset(
    scenario: Scenario,
    out_dir: str | Path,
    frames: str = "none",
    every: int = 600,
    progress: Callable[[int, int], None] | None = None,
    jobs: int = 1,
) -> dict[str, Any]:
    """Write one variant. frames: none (truth only, no rendering), sample, or all.

    ``sample`` renders and hashes every frame but stores a PNG only every `every` frames and at
    each change of a projector's geometry (a perturbation's onset or step). With `jobs` > 1 the
    frames are split into that many contiguous ranges rendered in parallel processes; every frame
    depends only on its index, so the files are the same as from one process.
    """
    if frames not in FRAME_MODES:
        raise ValueError(f"frames must be one of {FRAME_MODES}, got {frames!r}")
    if every < 1:
        raise ValueError(f"every must be at least 1, got {every}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    n = scenario.timing.n_frames
    scenario_text = yaml.safe_dump(scenario.data, sort_keys=False, allow_unicode=True)
    (out / "scenario.yaml").write_text(scenario_text)
    (out / "setup.json").write_text(dumps(scenario.setup_dict(), indent=1) + "\n")
    if frames != "none":
        (out / "frames").mkdir(exist_ok=True)
    jobs = max(1, min(jobs, n))
    if jobs == 1:
        render_s, written = _write_range(scenario, out, frames, every, 0, n, out / "metadata.jsonl", progress)
    else:
        bounds = [round(k * n / jobs) for k in range(jobs + 1)]
        parts = [out / f"metadata.part{k}.jsonl" for k in range(jobs)]
        with process_pool(jobs) as pool:
            futures = [pool.submit(_write_part, scenario.data, scenario.variant, str(out), frames, every,
                                   bounds[k], bounds[k + 1], str(parts[k])) for k in range(jobs)]
            results = [f.result() for f in futures]
        with open(out / "metadata.jsonl", "w") as fh:
            for part in parts:
                fh.write(part.read_text())
                part.unlink()
        render_s, written = sum(r[0] for r in results), sum(r[1] for r in results)
    total_s = time.perf_counter() - started
    info = {
        "scenario": scenario.name,
        "variant": scenario.variant,
        "scenario_sha256": hashlib.sha256(scenario_text.encode()).hexdigest(),
        "n_frames": n,
        "frames": frames,
        "pngs": written,
        "environment": environment(),
    }
    (out / "dataset.json").write_text(dumps(info, indent=1) + "\n")
    timing = {"total_s": round(total_s, 3), "render_s": round(render_s, 3), "jobs": jobs,
              "render_ms_per_frame": round(1000 * render_s / max(1, n), 2) if frames != "none" else None}
    (out / "timing.json").write_text(dumps(timing, indent=1) + "\n")
    return {"dir": str(out), **info, **timing}


@dataclass(eq=False)
class Dataset:
    """A dataset on disk. Frames come from the PNGs when present, else are re-rendered.

    Either way a frame is checked against its recorded hash, when the dataset has one.
    """

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
            png = self.path / line["png"]
            frame = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
            if frame is None:
                raise FileNotFoundError(f"{png}: frame {i}'s PNG is missing or unreadable")
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
