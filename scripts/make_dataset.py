"""Generate datasets from a scenario: ground truth, the setup the detector may read, optional frames.

Usage:
    python -m scripts.make_dataset scenarios/shift_sweep.yaml out/shift_sweep --frames sample --jobs 4

A scenario without a sweep is written to OUT itself; a sweep writes one directory per variant,
OUT/<variant>/, plus OUT/variants.json, the index of every variant written there so far.

  --frames none    truth only; nothing is rendered (seconds)
  --frames sample  every frame rendered and hashed; a 16-bit PNG stored every --every frames
                   (default 600: one per 5 minutes) and wherever a projector's geometry changes
  --frames all     every frame stored (about 8 MB each; refused above --max-gb)

Variants run in parallel with --jobs (a single variant splits its frames across the jobs
instead); the output does not depend on it. Rendering moves a lot of memory, so the gain is
machine-bound: sweeps of mostly static content run about twice as fast with 4 jobs on a 10-core
Mac, while video (new content every frame) is fastest in one process. Prints one JSON line per
variant, then a summary.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import time
from pathlib import Path

from sim.dataset import FRAME_MODES, dumps, process_pool, write_variant
from sim.render import QUALITY
from sim.scenario import load_scenarios

PNG_BYTES_PER_PIXEL = 1.1  # 16-bit PNG of a camera frame compresses to roughly this


def _at_least_one(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def _update_index(out: Path, scenario: str, order: list[str], written: list[str]) -> None:
    """Add the variants just written to OUT/variants.json, keeping those an earlier run wrote there.

    Variants are listed in the sweep's order; any the scenario no longer has go last.
    """
    path = out / "variants.json"
    known = [v["variant"] for v in json.loads(path.read_text())["variants"]] if path.exists() else []
    present = set(known) | set(written)
    names = [v for v in order if v in present] + [v for v in known if v not in order]
    index = {"scenario": scenario, "variants": [{"variant": v, "dir": v} for v in names]}
    path.write_text(dumps(index, indent=1) + "\n")


def main(argv: list[str] | None = None) -> list[dict]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenario", type=Path, help="scenario YAML file")
    parser.add_argument("out", type=Path, help="output directory")
    parser.add_argument("--frames", choices=FRAME_MODES, default="none", help="which frames to render and store")
    parser.add_argument("--quality", choices=sorted(QUALITY), default=None,
                        help="override the scenario's render quality preset")
    parser.add_argument("--variants", default=None, help="glob of variant names to write (default: all)")
    parser.add_argument("--jobs", type=_at_least_one, default=1, help="worker processes (default 1)")
    parser.add_argument("--every", type=_at_least_one, default=600, help="--frames sample: store every Nth frame")
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
    tasks = [(s.data, s.variant, str(args.out / s.variant if swept else args.out), args.frames, args.every)
             for s in chosen]
    started = time.perf_counter()
    if args.jobs > 1 and len(tasks) == 1:  # one variant: split its frames across the workers instead
        results = [write_variant(*tasks[0], jobs=args.jobs)]
    elif args.jobs > 1:
        with process_pool(args.jobs) as pool:
            results = list(pool.map(write_variant, *zip(*tasks, strict=True)))
    else:
        results = [write_variant(*task) for task in tasks]
    for r in results:
        print(json.dumps({k: r[k] for k in ("dir", "variant", "n_frames", "pngs", "total_s", "render_ms_per_frame")}))
    if swept:
        _update_index(args.out, scenarios[0].name, [s.variant for s in scenarios], [s.variant for s in chosen])
    print(json.dumps({"scenario": str(args.scenario), "variants": len(results), "wall_s": round(time.perf_counter() - started, 1)}))
    return results


if __name__ == "__main__":
    main()
