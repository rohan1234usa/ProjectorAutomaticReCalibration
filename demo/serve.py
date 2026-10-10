"""Serve the built site on this machine only (127.0.0.1), with caching off so a rebuild shows at once.

Needs only the standard library: ``python3 -m demo.serve [out/site] [--port 8000]`` serves an existing
build without the project's environment.
"""

from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - the base class's name
        pass


def serve(directory: Path, port: int = 8000) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, directory=str(directory)))
    print(f"Serving {directory} at http://localhost:{port}/  (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve the built demo site on localhost.")
    parser.add_argument("directory", nargs="?", type=Path, default=Path("out/site"))
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    serve(args.directory, args.port)
