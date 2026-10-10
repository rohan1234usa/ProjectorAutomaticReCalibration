"""The test suite, run for the demo and read back test by test.

pytest writes a JUnit XML report on its own (``--junitxml``), so no plugin is needed: one
``<testcase>`` per collected test with its time and, when it did not pass, the failure or skip
message. The test files themselves are read with ``ast`` (never imported) for what the report
lacks: each module's docstring (what the file is about), each test's docstring or, failing
that, its name in words, its source, and whether it is marked ``slow``.

The default build runs the suite as ``pytest`` does (``-m "not slow"``) and lists the slow tests
it left out; ``--tests all`` runs those too (they take minutes). The cache plugin is off so the
run writes nothing into the repository.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

MARKERS = {"fast": "not slow", "all": "slow or not slow"}

# Which part of the system each test file checks (a file not listed lands in "Other").
AREAS = {
    "Geometry and truth": ("test_planar", "test_arrangements", "test_perturb", "test_schedule", "test_truth"),
    "Light and blending": ("test_calibration", "test_render", "test_screen", "test_room"),
    "Camera and markers": ("test_camera", "test_fiducials"),
    "Content over time": ("test_content", "test_pictures", "test_sequence"),
    "Nuisances": ("test_nuisance",),
    "Scenarios and datasets": ("test_scenario", "test_catalogue", "test_frames", "test_dataset",
                               "test_check_dataset", "test_compare"),
    "Contracts and tools": ("test_config", "test_imports", "test_visualize", "test_demo", "test_demo_build"),
}


def area_of(module: str) -> str:
    return next((area for area, files in AREAS.items() if module in files), "Other")


def humanize(func: str) -> str:
    """test_overlap_adds_one_black_level -> Overlap adds one black level."""
    words = func.removeprefix("test_").replace("_", " ").strip()
    return words[:1].upper() + words[1:]


def _is_slow(decorator: ast.expr) -> bool:
    return "mark.slow" in ast.unparse(decorator)


def read_tests(tests_dir: Path) -> dict[str, dict[str, Any]]:
    """Per test module: its docstring and, per test function, docstring, source, line and slow flag."""
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(tests_dir.glob("test_*.py")):
        text = path.read_text()
        tree = ast.parse(text)
        lines = text.splitlines()
        funcs = {}
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                start = min([node.lineno] + [d.lineno for d in node.decorator_list])
                funcs[node.name] = {
                    "doc": ast.get_docstring(node) or "",
                    "source": "\n".join(lines[start - 1 : node.end_lineno]),
                    "line": node.lineno,
                    "slow": any(_is_slow(d) for d in node.decorator_list),
                    "parametrized": any("parametrize" in ast.unparse(d) for d in node.decorator_list),
                }
        out[path.stem] = {"doc": ast.get_docstring(tree) or "", "functions": funcs, "area": area_of(path.stem)}
    return out


def parse_junit(path: Path) -> list[dict[str, Any]]:
    """One record per test case: file, function, parameters, outcome, seconds, message."""
    cases = []
    for tc in ET.parse(path).getroot().iter("testcase"):
        classname, name = tc.get("classname", ""), tc.get("name", "")
        module = classname.split(".")[1] if classname.startswith("tests.") else classname.split(".")[-1]
        func, _, params = name.partition("[")
        outcome, message = "passed", ""
        for child in tc:
            if child.tag in ("failure", "error"):
                outcome = "failed" if child.tag == "failure" else "error"
                message = f"{child.get('message') or ''}\n{child.text or ''}".strip()
            elif child.tag == "skipped":
                outcome, message = "skipped", child.get("message") or ""
        cases.append({"file": module, "func": func, "params": params.removesuffix("]"), "outcome": outcome,
                      "seconds": round(float(tc.get("time") or 0.0), 4), "message": message[:4000]})
    return cases


def _pytest(repo: Path, *args: str) -> tuple[subprocess.CompletedProcess, float]:
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *args], cwd=repo,
                          capture_output=True, text=True)
    return proc, time.perf_counter() - t0


def run(repo: Path, junit: Path, mode: str) -> dict[str, Any]:
    """Run the suite (fast or all) writing `junit`; list the slow tests a fast run leaves out."""
    proc, wall = _pytest(repo, f"--junitxml={junit}", "-m", MARKERS[mode])
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    not_run: list[str] = []
    if mode == "fast":
        collected, _ = _pytest(repo, "--collect-only", "-m", "slow")  # addopts' -q already lists one test per line
        not_run = [line.strip() for line in collected.stdout.splitlines() if "::" in line]
    return {"mode": mode, "command": f"python -m pytest -m '{MARKERS[mode]}'", "returncode": proc.returncode,
            "python": sys.version.split()[0],
            "wall_s": round(wall, 1), "summary": lines[-1] if lines else "", "slow_not_run": not_run,
            "stderr_tail": proc.stderr[-2000:] if proc.returncode not in (0, 1) else ""}


def summarize(cases: list[dict[str, Any]], tests: dict[str, dict[str, Any]], run_info: dict[str, Any]) -> dict:
    """The tests page's data: totals, per area, per file, every case, the slowest."""
    outcomes = ("passed", "failed", "error", "skipped")
    totals = {k: sum(c["outcome"] == k for c in cases) for k in outcomes}
    files = []
    for module, info in tests.items():
        mine = [c for c in cases if c["file"] == module]
        files.append({"file": module, "area": info["area"], "doc": info["doc"], "count": len(mine),
                      "functions": len(info["functions"]),
                      "slow_functions": sum(f["slow"] for f in info["functions"].values()),
                      "seconds": round(sum(c["seconds"] for c in mine), 2),
                      **{k: sum(c["outcome"] == k for c in mine) for k in outcomes}})
    areas = []
    for area in [*AREAS, "Other"]:
        mine = [f for f in files if f["area"] == area]
        if mine:
            areas.append({"area": area, "files": [f["file"] for f in mine],
                          **{k: sum(f[k] for f in mine) for k in ("count", *outcomes)},
                          "seconds": round(sum(f["seconds"] for f in mine), 2)})
    for c in cases:
        fn = tests.get(c["file"], {}).get("functions", {}).get(c["func"], {})
        c["title"] = (fn.get("doc") or "").split("\n\n")[0].replace("\n", " ") or humanize(c["func"])
        c["slow"] = bool(fn.get("slow"))
    slowest = sorted(cases, key=lambda c: -c["seconds"])[:10]
    return {"run": run_info, "totals": {**totals, "total": len(cases), "slow_not_run": len(run_info["slow_not_run"]),
                                        "seconds": round(sum(c["seconds"] for c in cases), 1)},
            "areas": areas, "files": files, "cases": cases,
            "functions": {m: info["functions"] for m, info in tests.items()},
            "slowest": [{"file": c["file"], "func": c["func"], "params": c["params"], "seconds": c["seconds"]}
                        for c in slowest]}
