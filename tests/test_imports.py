"""Hard rules from CLAUDE.md section 3: detector/ imports nothing from sim/, and sim/ nothing from detector/.

Only the harness (eval/, scripts/) may use both.
"""

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


@pytest.mark.parametrize("package, forbidden", [("detector", "sim"), ("sim", "detector")])
def test_packages_stay_separate(package, forbidden):
    files = sorted((REPO / package).rglob("*.py"))
    assert files, f"{package} package not found"
    offenders = {
        str(f.relative_to(REPO)): sorted(m for m in _imported_modules(f) if m.split(".")[0] == forbidden)
        for f in files
    }
    assert not {f: m for f, m in offenders.items() if m}
