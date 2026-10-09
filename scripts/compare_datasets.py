"""Compare two dataset runs file by file: are they the same dataset?

The Phase 2 done conditions ask that a scenario "generates twice with identical metadata and
frame hashes". Rendering is deterministic (every frame depends only on its scenario and its
index), so two make_dataset runs on the same machine and OpenCV build must agree byte for byte
in everything but how long they took. This script checks that, and where two runs differ it
says where:

  variants         the same variants (a sweep root's variants.json and its directories)
  scenario.yaml    byte-identical; otherwise the first differing line
  setup.json       byte-identical; otherwise the first differing line
  dataset.json     identical but for its environment (commit, git_dirty, library versions,
                   platform), which is reported as information and never counts as a difference
  metadata.jsonl   line by line; a differing line is diffed field by field, so the report reads
                   "37 frames differ, the first is frame 74, in content.segments, frame_sha256"
  frames/*.png     the same files with the same bytes
  timing.json      ignored: how long a run took is not reproducible

``--ignore-fields`` drops metadata fields by dotted prefix, for runs whose metadata schema
differs on purpose (a dataset written before a field existed). A line whose text differs while
its parsed values agree (number formatting, key order) is reported under ``<format>``.

Usage:
    python -m scripts.compare_datasets out/run1/shift_sweep out/run2/shift_sweep
    python -m scripts.compare_datasets OLD NEW --ignore-fields content.frames_in_exposure,nuisances

Both paths are sweep roots (with variants.json) or both are single datasets. Prints one JSON
line per variant, then a summary, and exits non-zero if anything differs.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.dataset_files import file_sha256, variants


def _first_differing_line(a: bytes, b: bytes) -> int | None:
    la, lb = a.splitlines(), b.splitlines()
    for n, (x, y) in enumerate(zip(la, lb, strict=False), 1):
        if x != y:
            return n
    return None if len(la) == len(lb) else min(len(la), len(lb)) + 1


def compare_file(a: Path, b: Path) -> str:
    """"same", "missing in a", "missing in b" or "differs at line N"."""
    if not a.exists() or not b.exists():
        return "missing in a" if not a.exists() else "missing in b"
    line = _first_differing_line(a.read_bytes(), b.read_bytes())
    return "same" if line is None else f"differs at line {line}"


def diff_paths(x: Any, y: Any, prefix: str = "") -> list[str]:
    """Dotted paths at which two parsed JSON values differ; lists are compared whole."""
    if isinstance(x, dict) and isinstance(y, dict):
        out = []
        for key in sorted(set(x) | set(y)):
            path = f"{prefix}.{key}" if prefix else key
            out += [path] if key not in x or key not in y else diff_paths(x[key], y[key], path)
        return out
    return [] if x == y else [prefix or "<line>"]


def _ignored(path: str, ignore: tuple[str, ...]) -> bool:
    return any(path == f or path.startswith(f + ".") for f in ignore)


def compare_metadata(a: Path, b: Path, ignore: tuple[str, ...], max_list: int) -> dict[str, Any]:
    """Line-by-line comparison of two metadata.jsonl files, diffing differing lines field by field."""
    la, lb = a.read_text().splitlines(), b.read_text().splitlines()
    frames, fields, first = [], Counter(), None
    for n, (x, y) in enumerate(zip(la, lb, strict=False)):
        if x == y:
            continue
        try:
            dx, dy = json.loads(x), json.loads(y)
        except json.JSONDecodeError:
            paths = ["<unparseable>"]
        else:
            raw = diff_paths(dx, dy)
            paths = [p for p in raw if not _ignored(p, ignore)] if raw else ["<format>"]
        if not paths:
            continue  # only ignored fields differ
        frames.append(n)
        fields.update(paths)
        if first is None:
            first = {"i": n, "fields": paths}
    return {"lines": [len(la), len(lb)], "differing_frames": len(frames), "first": first,
            "fields": dict(sorted(fields.items())), "frames": frames[:max_list],
            "truncated": len(frames) > max_list,
            "same": len(la) == len(lb) and not frames}


def compare_pngs(a: Path, b: Path) -> dict[str, Any]:
    """The stored frames: the same files under frames/, with the same bytes."""
    def listing(root: Path) -> set[str]:
        return {str(p.relative_to(root)) for p in (root / "frames").glob("*.png")} if (root / "frames").exists() else set()

    pa, pb = listing(a), listing(b)
    common = sorted(pa & pb)
    differing = [p for p in common if file_sha256(a / p) != file_sha256(b / p)]
    return {"same_files": len(common) - len(differing), "only_in_a": sorted(pa - pb), "only_in_b": sorted(pb - pa),
            "differing": differing, "same": pa == pb and not differing}


def compare_info(a: Path, b: Path) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    """dataset.json: (keys that differ outside the environment, environment differences as information)."""
    if not (a / "dataset.json").exists() or not (b / "dataset.json").exists():
        return {"dataset.json": "missing in a" if not (a / "dataset.json").exists() else "missing in b"}, {}
    ia, ib = json.loads((a / "dataset.json").read_text()), json.loads((b / "dataset.json").read_text())
    ea, eb = ia.pop("environment", {}), ib.pop("environment", {})
    differs = {k: [ia.get(k), ib.get(k)] for k in sorted(set(ia) | set(ib)) if ia.get(k) != ib.get(k)}
    env = {k: [ea.get(k), eb.get(k)] for k in sorted(set(ea) | set(eb)) if ea.get(k) != eb.get(k)}
    return differs, env


def compare_variant(a: Path, b: Path, ignore: tuple[str, ...] = (), max_list: int = 100) -> dict[str, Any]:
    files = {name: compare_file(a / name, b / name) for name in ("scenario.yaml", "setup.json")}
    info, environment = compare_info(a, b)
    metadata = (compare_metadata(a / "metadata.jsonl", b / "metadata.jsonl", ignore, max_list)
                if (a / "metadata.jsonl").exists() and (b / "metadata.jsonl").exists()
                else {"same": False, "missing": "a" if not (a / "metadata.jsonl").exists() else "b"})
    pngs = compare_pngs(a, b)
    same = all(v == "same" for v in files.values()) and not info and metadata["same"] and pngs["same"]
    return {"a": str(a), "b": str(b), "same": same, "files": files, "dataset_json": info,
            "environment": environment, "metadata": metadata, "pngs": pngs}


def compare(a: Path, b: Path, ignore: tuple[str, ...] = (), max_list: int = 100) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compare two sweep roots or two single datasets: (one result per variant, a summary)."""
    va, vb = variants(a), variants(b)
    if (va is None) != (vb is None):
        raise ValueError(f"{a if va is not None else b} is a sweep root (variants.json) and the other is not")
    if va is None:
        results = [{"variant": a.name, **compare_variant(a, b, ignore, max_list)}]
        summary: dict[str, Any] = {}
    else:
        index = compare_file(a / "variants.json", b / "variants.json")
        results = [{"variant": v, **compare_variant(a / v, b / v, ignore, max_list)}
                   for v in va if v in vb]
        summary = {"variants.json": index, "only_in_a": [v for v in va if v not in vb],
                   "only_in_b": [v for v in vb if v not in va]}
    differing = [r["variant"] for r in results if not r["same"]]
    ok = not differing and summary.get("variants.json", "same") == "same" \
        and not summary.get("only_in_a") and not summary.get("only_in_b")
    return results, {"ok": ok, "variants": len(results), "differing": len(differing), **summary}


def main(argv: list[str] | None = None) -> bool:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("a", type=Path, help="a dataset directory or sweep root")
    parser.add_argument("b", type=Path, help="the dataset directory or sweep root to compare it with")
    parser.add_argument("--ignore-fields", default="", help="comma-separated dotted metadata fields to ignore")
    parser.add_argument("--max-list", type=int, default=100, help="list at most this many differing frames")
    args = parser.parse_args(argv)
    ignore = tuple(f.strip() for f in args.ignore_fields.split(",") if f.strip())
    try:
        results, summary = compare(args.a, args.b, ignore, args.max_list)
    except ValueError as error:
        parser.error(str(error))
    for r in results:
        print(json.dumps(r))
    print(json.dumps(summary))
    return summary["ok"]


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
