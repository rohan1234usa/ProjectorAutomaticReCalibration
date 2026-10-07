"""Hard rule from CLAUDE.md section 3: detector/ imports nothing from sim/."""

import ast
from pathlib import Path

DETECTOR = Path(__file__).resolve().parents[1] / "detector"


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def test_detector_never_imports_sim():
    files = sorted(DETECTOR.rglob("*.py"))
    assert files, "detector package not found"
    offenders = {
        str(f.relative_to(DETECTOR.parent)): sorted(m for m in _imported_modules(f) if m.split(".")[0] == "sim")
        for f in files
    }
    assert not {f: m for f, m in offenders.items() if m}
