"""Assemble the site: the static pages, the generated data files, and a safe swap into place.

The pages in ``demo/site`` are written by hand; everything that comes from the code, the
documents or the simulator goes into ``data/*.js`` as ``window.DEMO.<name> = {...}``, which a page
loads with a plain script tag -- so the site also opens straight from disk, no server needed. Each
page loads only its own data: ``common`` (the documents) plus ``overview``, ``samples``,
``algorithm`` or ``tests``. The four pages share one header, kept once in ``site/_header.html``.

Every data file carries its own provenance (``meta``: when it was made, from which commit, at
which quality), because a build can reuse parts of the last one: ``--samples skip`` and
``--tests skip`` carry the previous rendered images, sample data and test results over, for quick
work on the pages, and the pages then say when and from what those parts were made. Data from an
older layout (another ``SCHEMA``) is not reused; with nothing to carry over, that part's data file
holds ``null`` and the pages say it was not built.

The build happens in a temporary directory next to the target and replaces it only when it has
finished, so a failed build leaves the last good site in place. It replaces only a directory it
made itself (the ``.demo-site`` marker file), or an empty one.
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
HEADER = "_header.html"  # the shared page header, put in place of <!-- header --> on every page
MARKER = ".demo-site"
REPO = sources.REPO
SCHEMA = 2  # the data files' layout; data written under another one is never reused
_PREFIX = "window.DEMO = window.DEMO || {};\n"


def _plain(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def write_data(root: Path, name: str, obj: Any) -> int:
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False, default=_plain)
    path = root / "data" / f"{name}.js"
    path.write_text(f"{_PREFIX}window.DEMO[{json.dumps(name)}] = {text};\n")
    return path.stat().st_size


def read_data(root: Path, name: str) -> dict[str, Any] | None:
    """A data file written by :func:`write_data`, read back (None if missing or not in that form)."""
    path = root / "data" / f"{name}.js"
    head = f"{_PREFIX}window.DEMO[{json.dumps(name)}] = "
    try:
        text = path.read_text()
    except OSError:
        return None
    if not (text.startswith(head) and text.endswith(";\n")):
        return None
    try:
        return json.loads(text[len(head) : -2])
    except ValueError:
        return None


def check_target(out: Path) -> None:
    if out.exists() and (not out.is_dir() or (any(out.iterdir()) and not (out / MARKER).exists())):
        raise SystemExit(f"{out} exists and was not made by make_site (no {MARKER} file): refusing to replace it")


def _reuse(out: Path, tmp: Path, names: list[str], extra: list[str], what: str, log: Any) -> dict[str, Any] | None:
    """Carry the last build's data files (and `extra` paths) over, all or nothing; None if they can't be."""
    data = {name: read_data(out, name) for name in names}
    stale = [n for n, d in data.items() if d is None or d.get("meta", {}).get("schema") != SCHEMA]
    missing = stale + [rel for rel in extra if not (out / rel).exists()]
    if missing:
        log(f"--{what} skip: nothing to reuse ({', '.join(missing)} missing or from an older build); the pages will say so")
        return None
    for rel in [f"data/{n}.js" for n in names] + extra:
        src, dst = out / rel, tmp / rel
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
    return data


def _tests(tmp: Path, mode: str, log: Any) -> dict[str, Any]:
    junit = tmp / "data" / "junit.xml"
    log(f"running the test suite ({mode}); this takes a minute or two")
    info = testrun.run(REPO, junit, mode)
    log(f"  {info['summary']} ({info['wall_s']} s)")
    if not junit.exists():
        raise SystemExit(f"pytest wrote no report (exit {info['returncode']}):\n{info['stderr_tail']}")
    return testrun.summarize(testrun.parse_junit(junit), testrun.read_tests(REPO / "tests"), info)


def overview(samples: dict[str, Any] | None, tests: dict[str, Any] | None, meta: dict[str, Any]) -> dict[str, Any]:
    """The few numbers and pictures the overview page shows, so it need not load the others' data."""
    out: dict[str, Any] = {"meta": meta, "tests": None, "hero": None}
    if tests is not None:
        out["tests"] = {k: tests["totals"][k] for k in ("passed", "failed", "error", "total", "slow_not_run")}
    if samples is not None:
        shift = samples["shift"]
        out["hero"] = {"frame": shift["frame"], "sizes_px": [s["size_px"] for s in shift["steps"]], **shift["hero"],
                       "aligned": shift["steps"][0]["crops"][0]}
    return out


def _pages(tmp: Path) -> None:
    """Put the shared header into every page."""
    header = (tmp / HEADER).read_text()
    (tmp / HEADER).unlink()
    for page in tmp.glob("*.html"):
        text = page.read_text()
        if "<!-- header -->" not in text:
            raise SystemExit(f"{page.name}: no <!-- header --> to replace")
        page.write_text(text.replace("<!-- header -->", header.strip()))


def build(out: Path, samples: str = "standard", tests: str = "fast", log: Any = None) -> dict[str, Any]:
    """Build the site into `out`; returns a short summary."""
    log = log or (lambda msg: print(msg, file=sys.stderr, flush=True))
    out = out.resolve()
    check_target(out)
    tmp = out.parent / f".{out.name}.building-{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(STATIC, tmp)
    _pages(tmp)
    (tmp / "data").mkdir(exist_ok=True)
    (tmp / "img").mkdir(exist_ok=True)
    t0 = time.perf_counter()
    env = sources.environment()

    def meta(**extra: Any) -> dict[str, Any]:
        return {"schema": SCHEMA, "built": env["built"], "commit": env["short"], "dirty": env["dirty"], **extra}

    try:
        if tests == "skip":
            reused = _reuse(out, tmp, ["tests"], ["data/junit.xml"], "tests", log)
            test_data = reused["tests"] if reused else None
            if test_data is None:
                write_data(tmp, "tests", None)  # every page's scripts exist; this one says "not built"
        else:
            test_data = {**_tests(tmp, tests, log), "meta": meta(mode=tests)}
            write_data(tmp, "tests", test_data)
        common = {**sources.common(), "meta": meta()}
        if samples == "skip":
            reused = _reuse(out, tmp, ["samples", "algorithm"], ["img"], "samples", log)
            sample_data = reused["samples"] if reused else None
            if sample_data is None:
                write_data(tmp, "samples", None)
                write_data(tmp, "algorithm", None)
        else:
            log(f"rendering samples at {samples} quality")
            sample_data, algorithm = sample_figures.render(tmp, samples, common["config"]["values"], log)
            sample_data["meta"], algorithm["meta"] = meta(quality=samples), meta(quality=samples)
            write_data(tmp, "samples", sample_data)
            write_data(tmp, "algorithm", algorithm)
        write_data(tmp, "overview", overview(sample_data, test_data, meta()))
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
