"""
tools/serve_ui.py — Serves the browser test UI (static/index.html) on localhost.

Usage:
    python tools/serve_ui.py           # http://localhost:3000
    python tools/serve_ui.py --port 8888

Open http://localhost:3000 in your browser to access the voice agent test UI.
Make sure the token server (token_server.py) is running on port 8080 first.
"""

import argparse
import http.server
import logging
import os
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent.parent / "static"


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def log_message(self, fmt, *args):
        logger.info(fmt % args)


def main():
    parser = argparse.ArgumentParser(description="Serve the browser test UI")
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument("--no-open", action="store_true", help="Don't auto-open browser")
    args = parser.parse_args()

    url = f"http://localhost:{args.port}"
    logger.info(f"Serving UI at {url}")
    logger.info("Make sure the token server is running: python token_server.py")

    if not args.no_open:
        import threading
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    server = http.server.HTTPServer(("0.0.0.0", args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("UI server stopped.")


if __name__ == "__main__":
    main()
