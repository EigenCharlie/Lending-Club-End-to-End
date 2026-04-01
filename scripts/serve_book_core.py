"""Serve the frozen book locally and open it in a browser.

This avoids Quarto preview execution, which can still touch live kernels in
the user's Jupyter environment. The book content is served from compiled
static output in either the frozen core or the full book.

Usage:
    uv run python scripts/serve_book_core.py
    uv run python scripts/serve_book_core.py --book core
    uv run python scripts/serve_book_core.py --book full --port 8001 --no-open
"""

from __future__ import annotations

import argparse
import pathlib
import threading
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
BOOK_CORE_DIR = REPO_ROOT / "book-core"
BOOK_FULL_DIR = REPO_ROOT / "book"


def _resolve_output_dir(book_mode: str) -> pathlib.Path:
    if book_mode == "core":
        return BOOK_CORE_DIR / "_output-core"
    return BOOK_FULL_DIR / "_output"


def _ensure_output_dir(output_dir: pathlib.Path) -> None:
    index_html = output_dir / "index.html"
    if index_html.exists():
        return
    raise FileNotFoundError(
        f"Frozen output not found at {output_dir}. Render the selected book first."
    )


def _open_browser(url: str) -> None:
    import contextlib

    with contextlib.suppress(Exception):
        webbrowser.open(url, new=2, autoraise=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--book",
        choices=("core", "full"),
        default="full",
        help="Which compiled book to serve.",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not attempt to open a browser automatically.",
    )
    args = parser.parse_args()

    output_dir = _resolve_output_dir(args.book)
    _ensure_output_dir(output_dir)

    handler = partial(SimpleHTTPRequestHandler, directory=str(output_dir))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}/"

    print(f"Serving frozen book from: {output_dir}")
    print(f"Open in browser: {url}")

    if not args.no_open:
        threading.Timer(0.5, _open_browser, args=(url,)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
