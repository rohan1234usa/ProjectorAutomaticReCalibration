"""What the project's own documents say, read at build time so the pages never drift from them.

- CLAUDE.md section 8 (build order): each phase's scope, done condition and measured status.
  A phase is *done* when its status cell starts with "done"; the first phase that is not is
  *next*.
- CLAUDE.md section 7 (scenario catalogue): the question each scenario answers. The scenario
  files themselves give the variant and frame counts.
- detector.yaml: every tunable with its value and comment, grouped under the comment headers.
  It is read as text, because the comments are the explanation and YAML loaders drop them;
  values still go through ``yaml.safe_load`` so they are the numbers the detector will read.
- docs/findings.md: one entry per dated ``##`` heading.
- The build environment: git commit, whether the tree had uncommitted changes, versions.
"""

from __future__ import annotations

import datetime as dt
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from demo.markdown import inline, table_rows, to_html
from sim.scenario import load_scenarios

REPO = Path(__file__).resolve().parents[1]


def section(text: str, number: str) -> list[str]:
    """Lines of the ``## <number>.`` section of a Markdown document."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(f"## {number}."))
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    return lines[start + 1 : end]


def first_table(lines: list[str]) -> list[list[str]]:
    start = next(i for i, line in enumerate(lines) if line.lstrip().startswith("|"))
    end = next((i for i in range(start, len(lines)) if not lines[i].lstrip().startswith("|")), len(lines))
    return table_rows(lines[start:end])


def phases(claude_md: str) -> list[dict[str, Any]]:
    """The build-order table, with a state per phase: done, next or later."""
    rows = first_table(section(claude_md, "8"))
    header, out, seen_next = rows[0], [], False
    if [h.lower() for h in header] != ["phase", "scope", "done condition", "depends on", "status"]:
        raise ValueError(f"CLAUDE.md section 8: unexpected table header {header}")
    for cells in rows[1:]:
        if len(cells) != 5:
            raise ValueError(f"CLAUDE.md section 8: a row has {len(cells)} cells, expected 5: {cells[0]}")
        phase, scope, done, depends, status = cells
        if status.lstrip("*").lower().startswith("done"):
            state = "done"
        else:
            state, seen_next = ("later", True) if seen_next else ("next", True)
        number, _, title = phase.partition(" ")
        out.append({"number": number, "title": title, "scope": inline(scope), "done_condition": inline(done),
                    "depends": depends, "status": inline(status), "state": state})
    return out


def scenario_files(scenarios: Path) -> dict[str, list]:
    """Every scenario file's variants, by file stem (bases such as _lecture_hall.yaml skipped)."""
    return {p.stem: load_scenarios(p) for p in sorted(scenarios.glob("*.yaml")) if not p.stem.startswith("_")}


def catalogue(claude_md: str, files: dict[str, list]) -> list[dict[str, Any]]:
    """Each catalogue scenario with its question and its counts from the scenario file."""
    out = []
    for name, question in first_table(section(claude_md, "7"))[1:]:
        stem = name.strip("`")
        variants = files[stem]
        timing = variants[0].timing
        out.append({"name": stem, "question": inline(question), "variants": len(variants),
                    "frames_per_variant": timing.n_frames, "frames": sum(v.timing.n_frames for v in variants),
                    "duration_s": float(timing.duration), "reference": bool(variants[0].reference["available"])})
    return out


def config(text: str) -> dict[str, Any]:
    """detector.yaml as sections of {key, value, comment}, plus a flat key -> value map."""
    sections: list[dict[str, Any]] = []
    values: dict[str, Any] = {}
    pending_header = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            pending_header = None
            continue
        if line.startswith("#"):
            pending_header = line.lstrip("#").strip()
            continue
        body, _, comment = raw.partition("#")
        key, _, value = body.partition(":")
        key = key.strip()
        if pending_header is not None:
            sections.append({"name": pending_header, "items": []})
            pending_header = None
        if not sections:
            raise ValueError(f"detector.yaml: {key} sits under no section header")
        parsed = yaml.safe_load(value)
        values[key] = parsed
        sections[-1]["items"].append({"key": key, "value": parsed, "comment": comment.strip()})
    return {"sections": sections, "values": values}


def findings(text: str) -> list[dict[str, str]]:
    """The dated entries of docs/findings.md, oldest first, each as HTML."""
    entries, current = [], None
    for line in text.splitlines():
        if line.startswith("## "):
            heading = line[3:].strip()
            date, _, title = heading.partition(" — ")
            current = {"date": date.strip(), "title": title.strip() or heading, "lines": []}
            entries.append(current)
        elif current is not None:
            current["lines"].append(line)
    out = []
    for n, e in enumerate(entries, start=1):
        body = to_html("\n".join(e["lines"]))
        body = re.sub(r'<h(\d) id="([^"]*)">', lambda m: f'<h{m.group(1)} id="f{n}-{m.group(2)}">', body)
        body = body.replace(f'id="f{n}-numbers">', f'id="f{n}-numbers" class="numbers">')
        out.append({"date": e["date"], "title": inline(e["title"]), "html": body, "anchor": f"finding-{n}"})
    return out


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True, cwd=REPO).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment() -> dict[str, Any]:
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "built": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "commit": commit,
        "short": commit[:7] if commit else None,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
        "subject": _git("log", "-1", "--format=%s"),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "platform": platform.platform(terse=True),
    }


def common(repo: Path = REPO) -> dict[str, Any]:
    """Everything the pages take from the documents (cheap: rebuilt on every build)."""
    claude_md = (repo / "CLAUDE.md").read_text()
    files = scenario_files(repo / "scenarios")
    return {
        "env": environment(),
        "phases": phases(claude_md),
        "catalogue": catalogue(claude_md, files),
        "counts": {"scenarios": len(files), "variants": sum(len(v) for v in files.values()),
                   "frames": sum(s.timing.n_frames for v in files.values() for s in v)},
        "config": config((repo / "detector.yaml").read_text()),
        "findings": findings((repo / "docs" / "findings.md").read_text()),
    }
