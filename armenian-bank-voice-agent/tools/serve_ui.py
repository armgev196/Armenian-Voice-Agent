import argparse
import http.server
import logging
import sys
import threading
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent.parent / "static"


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def log_message(self, fmt, *args) -> None:
        logger.debug(fmt, *args)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    url = f"http://localhost:{args.port}"
    server = http.server.HTTPServer(("0.0.0.0", args.port), Handler)
    logger.info("UI at %s  (token server expected on :8080)", url)

    if not args.no_open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
