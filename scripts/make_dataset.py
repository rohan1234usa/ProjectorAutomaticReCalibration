"""Generate datasets from a scenario: ground truth, the setup the detector may read, optional frames.

Usage:
    python -m scripts.make_dataset scenarios/shift_sweep.yaml out/shift_sweep --frames sample --jobs 4

A scenario without a sweep is written to OUT itself; a sweep writes one directory per variant,
OUT/<variant>/, plus OUT/variants.json.

  --frames none    truth only; nothing is rendered (seconds)
  --frames sample  every frame rendered and hashed; a 16-bit PNG stored every --every frames
                   (default 600: one per 5 minutes) and wherever a projector's geometry changes
  --frames all     every frame stored (about 8 MB each; refused above --max-gb)

Variants run in parallel with --jobs; the output does not depend on it. Prints one JSON line
per variant, then a summary.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2

from sim.dataset import FRAME_MODES, dumps, write_dataset
from sim.render import QUALITY
from sim.scenario import load_scenario, load_scenarios

PNG_BYTES_PER_PIXEL = 1.1  # 16-bit PNG of a camera frame compresses to roughly this


def _init_worker(threads: int) -> None:
    cv2.setNumThreads(threads)


def _write(scenario_path: str, quality: str | None, variant: str, out_dir: str, frames: str, every: int) -> dict:
    scenario = load_scenario(scenario_path, quality=quality, variant=variant)
    return write_dataset(scenario, out_dir, frames=frames, every=every)


def main(argv: list[str] | None = None) -> list[dict]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scenario", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--frames", choices=FRAME_MODES, default="none")
    parser.add_argument("--quality", choices=sorted(QUALITY), default=None)
    parser.add_argument("--variants", default=None, help="glob of variant names to write (default: all)")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--every", type=int, default=600, help="--frames sample: store every Nth frame")
    parser.add_argument("--max-gb", type=float, default=5.0, help="--frames all: refuse above this estimate")
    args = parser.parse_args(argv)

    scenarios = load_scenarios(args.scenario, quality=args.quality)
    chosen = [s for s in scenarios if args.variants is None or fnmatch.fnmatch(s.variant, args.variants)]
    if not chosen:
        raise SystemExit(f"no variant matches {args.variants!r}; variants: {[s.variant for s in scenarios]}")
    if args.frames == "all":
        gb = sum(s.timing.n_frames * s.scene.camera.resolution[0] * s.scene.camera.resolution[1] for s in chosen)
        gb *= PNG_BYTES_PER_PIXEL / 1e9
        if gb > args.max_gb:
            raise SystemExit(f"--frames all would write about {gb:.0f} GB; use --frames sample or raise --max-gb")
    swept = len(scenarios) > 1
    jobs = [(str(args.scenario), args.quality, s.variant, str(args.out / s.variant if swept else args.out),
             args.frames, args.every) for s in chosen]
    started = time.perf_counter()
    if args.jobs > 1 and len(jobs) > 1:
        threads = max(1, (os.cpu_count() or 1) // args.jobs)
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(args.jobs, mp_context=context, initializer=_init_worker, initargs=(threads,)) as pool:
            results = list(pool.map(_write, *zip(*jobs)))
    else:
        results = [_write(*job) for job in jobs]
    for r in results:
        print(json.dumps({k: r[k] for k in ("dir", "variant", "n_frames", "pngs", "total_s", "render_ms_per_frame")}))
    if swept:
        index = {"scenario": scenarios[0].name, "variants": [{"variant": s.variant, "dir": s.variant} for s in scenarios]}
        (args.out / "variants.json").write_text(dumps(index, indent=1) + "\n")
    print(json.dumps({"scenario": str(args.scenario), "variants": len(results), "wall_s": round(time.perf_counter() - started, 1)}))
    return results


if __name__ == "__main__":
    main()
