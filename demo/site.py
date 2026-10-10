"""Assemble the site: the static pages, the generated data files, and a safe swap into place.

The pages in ``demo/site`` are written by hand; everything that comes from the code, the
documents or the simulator goes into ``data/*.js`` as ``window.DEMO.<name> = {...}``, which a page
loads with a plain script tag -- so the site also opens straight from disk, no server needed.

The build happens in a temporary directory next to the target and replaces it only when it has
finished, so a failed build leaves the last good site in place. It replaces only a directory it
made itself (the ``.demo-site`` marker file), or an empty one. ``--samples skip`` and
``--tests skip`` carry the previous build's rendered images, sample data and test results over,
for quick work on the pages; with nothing to carry over, the pages say that part was not built.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from demo import samples as sample_figures
from demo import sources, testrun

STATIC = Path(__file__).resolve().parent / "site"
MARKER = ".demo-site"
REPO = sources.REPO


def _plain(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def write_data(root: Path, name: str, obj: Any) -> int:
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False, default=_plain)
    path = root / "data" / f"{name}.js"
    path.write_text(f"window.DEMO = window.DEMO || {{}};\nwindow.DEMO[{json.dumps(name)}] = {text};\n")
    return path.stat().st_size


def check_target(out: Path) -> None:
    if out.exists() and (not out.is_dir() or (any(out.iterdir()) and not (out / MARKER).exists())):
        raise SystemExit(f"{out} exists and was not made by make_site (no {MARKER} file): refusing to replace it")


def _reuse(out: Path, tmp: Path, paths: list[str], what: str, log: Any) -> None:
    """Carry the last build's files over; without them the pages say that part was not built."""
    missing = [rel for rel in paths if not (out / rel).exists()]
    if missing:
        log(f"--{what} skip: nothing to reuse ({', '.join(missing)} missing); the pages will say so")
        return
    for rel in paths:
        src, dst = out / rel, tmp / rel
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)


def _tests(tmp: Path, mode: str, log: Any) -> dict[str, Any]:
    junit = tmp / "data" / "junit.xml"
    log(f"running the test suite ({mode}); this takes a minute or two")
    info = testrun.run(REPO, junit, mode)
    log(f"  {info['summary']} ({info['wall_s']} s)")
    if not junit.exists():
        raise SystemExit(f"pytest wrote no report (exit {info['returncode']}):\n{info['stderr_tail']}")
    return testrun.summarize(testrun.parse_junit(junit), testrun.read_tests(REPO / "tests"), info)


def build(out: Path, samples: str = "standard", tests: str = "fast", log: Any = None) -> dict[str, Any]:
    """Build the site into `out`; returns a short summary."""
    log = log or (lambda msg: print(msg, file=sys.stderr, flush=True))
    out = out.resolve()
    check_target(out)
    tmp = out.parent / f".{out.name}.building-{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(STATIC, tmp)
    (tmp / "data").mkdir(exist_ok=True)
    (tmp / "img").mkdir(exist_ok=True)
    t0 = time.perf_counter()
    try:
        if tests == "skip":
            _reuse(out, tmp, ["data/tests.js", "data/junit.xml"], "tests", log)
        else:
            write_data(tmp, "tests", _tests(tmp, tests, log))
        common = sources.common()
        common["build"] = {"samples": samples, "tests": tests}
        if samples == "skip":
            _reuse(out, tmp, ["img", "data/samples.js", "data/algorithm.js"], "samples", log)
        else:
            log(f"rendering samples at {samples} quality")
            data, algorithm = sample_figures.render(tmp, samples, common["config"]["values"], log)
            write_data(tmp, "samples", data)
            write_data(tmp, "algorithm", algorithm)
        write_data(tmp, "common", common)
        (tmp / MARKER).write_text("Made by `python -m scripts.make_site`; replaced on every build.\n")
        old = out.parent / f".{out.name}.old-{os.getpid()}"
        if out.exists():
            out.rename(old)
        tmp.rename(out)
        shutil.rmtree(old, ignore_errors=True)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    summary = {"out": str(out), "seconds": round(time.perf_counter() - t0, 1), "megabytes": round(size / 1e6, 1),
               "images": len(list((out / "img").glob("*")))}
    log(f"site built in {summary['seconds']} s: {summary['images']} images, {summary['megabytes']} MB in {out}")
    return summary
