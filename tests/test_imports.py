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


@pytest.mark.parametrize("script", ["check_dataset", "check_geometry", "check_timeline", "check_frames",
                                    "dataset_files", "compare_datasets"])
def test_the_dataset_checkers_keep_their_own_code(script):
    """The checkers re-derive the truth apart from the simulator, so its bugs cannot pass its own check.

    Only re-rendering (scripts/check_frames.py) runs the simulator, importing it inside that function.
    """
    tree = ast.parse((REPO / "scripts" / f"{script}.py").read_text())
    top = {alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names}
    top |= {node.module for node in tree.body if isinstance(node, ast.ImportFrom) and node.module}
    assert not sorted(m for m in top if m.split(".")[0] == "sim")
