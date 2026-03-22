import argparse
import base64
import hashlib
import hmac
import json
import logging
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from config import LIVEKIT_API_KEY, LIVEKIT_API_SECRET, LIVEKIT_URL

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def create_livekit_token(
    api_key: str,
    api_secret: str,
    room: str,
    identity: str,
    ttl_seconds: int = 3600,
    can_publish: bool = True,
    can_subscribe: bool = True,
) -> str:
    now = int(time.time())
    header = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    payload = _b64url(json.dumps({
        "iss": api_key,
        "sub": identity,
        "iat": now,
        "nbf": now,
        "exp": now + ttl_seconds,
        "jti": f"{identity}-{now}",
        "video": {
            "room": room,
            "roomJoin": True,
            "canPublish": can_publish,
            "canSubscribe": can_subscribe,
        },
    }, separators=(",", ":")).encode())

    sig = _b64url(
        hmac.new(api_secret.encode(), f"{header}.{payload}".encode(), hashlib.sha256).digest()
    )
    return f"{header}.{payload}.{sig}"


class TokenHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args) -> None:
        logger.info(fmt, *args)

    def _json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if parsed.path == "/health":
            self._json({"status": "ok"})

        elif parsed.path == "/token":
            room = params.get("room", ["bank-support"])[0]
            identity = params.get("identity", [f"user-{int(time.time())}"])[0]
            token = create_livekit_token(
                api_key=LIVEKIT_API_KEY,
                api_secret=LIVEKIT_API_SECRET,
                room=room,
                identity=identity,
            )
            self._json({"token": token, "url": LIVEKIT_URL, "room": room, "identity": identity})
            logger.info("Token issued — room=%s identity=%s", room, identity)

        else:
            self._json({"error": "not found"}, status=404)


def run(port: int = 8080) -> None:
    server = HTTPServer(("0.0.0.0", port), TokenHandler)
    logger.info("Token server on http://localhost:%d", port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    run(port=parser.parse_args().port)
