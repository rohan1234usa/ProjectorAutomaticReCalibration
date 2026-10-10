"""Build the demo site -- sample data, the planned algorithm, the tests so far -- and serve it on localhost.

The build renders sample frames from the scenarios (about a minute at standard quality), runs the
test suite (``-m "not slow"`` unless ``--tests all``), reads CLAUDE.md, docs/findings.md and
detector.yaml, and writes a static site. ``--serve`` then serves it at http://localhost:8000.

Usage:
    python -m scripts.make_site                          # build out/site: standard renders, fast tests
    python -m scripts.make_site --serve                  # ... and serve it on localhost
    python -m scripts.make_site --samples fast --tests all            # quicker renders; slow tests too
    python -m scripts.make_site --samples skip --tests skip --serve   # refresh the pages only, keep the data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from demo.serve import serve
from demo.site import build


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", nargs="?", type=Path, default=Path("out/site"), help="output directory (default out/site)")
    parser.add_argument("--samples", choices=("standard", "fast", "skip"), default="standard",
                        help="render quality of the sample frames, or skip to keep the last build's")
    parser.add_argument("--tests", choices=("fast", "all", "skip"), default="fast",
                        help="run the default suite, all tests including slow ones, or keep the last results")
    parser.add_argument("--serve", action="store_true", help="serve the site on localhost after building it")
    parser.add_argument("--port", type=int, default=8000, help="port for --serve (default 8000)")
    args = parser.parse_args(argv)
    summary = build(args.out, samples=args.samples, tests=args.tests)
    if args.serve:
        serve(args.out, args.port)
    return summary


if __name__ == "__main__":
    main(sys.argv[1:])
