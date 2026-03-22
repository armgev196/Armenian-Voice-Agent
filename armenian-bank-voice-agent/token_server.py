"""
token_server.py — Lightweight HTTP server that issues LiveKit access tokens.

The browser-based test client (static/index.html) calls this server to get a
signed JWT token before connecting to the LiveKit room.  This is the standard
pattern for any LiveKit deployment (cloud or self-hosted).

Usage:
    python token_server.py           # runs on http://localhost:8080
    python token_server.py --port 9000

Why a separate token server?
  - LiveKit tokens must be signed server-side with the API secret
  - You never expose the API secret to the browser
  - This server is intentionally minimal (stdlib only, no framework dep)
"""

import argparse
import json
import logging
import time
import hmac
import hashlib
import base64
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

from config import LIVEKIT_API_KEY, LIVEKIT_API_SECRET

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Minimal LiveKit JWT implementation (no external JWT library needed) ────────

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
    """
    Create a signed LiveKit access token (JWT).
    Implements the LiveKit token spec without external JWT dependencies.
    """
    now = int(time.time())

    header = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())

    payload = {
        "iss": api_key,
        "sub": identity,
        "iat": now,
        "nbf": now,
        "exp": now + ttl_seconds,
        "jti": f"{identity}-{now}",
        "video": {
            "room":         room,
            "roomJoin":     True,
            "canPublish":   can_publish,
            "canSubscribe": can_subscribe,
        },
    }
    payload_b64 = _b64url(json.dumps(payload, separators=(",", ":")).encode())

    signing_input = f"{header}.{payload_b64}".encode()
    sig = hmac.new(api_secret.encode(), signing_input, hashlib.sha256).digest()
    signature = _b64url(sig)

    return f"{header}.{payload_b64}.{signature}"


# ── HTTP Request Handler ───────────────────────────────────────────────────────

class TokenHandler(BaseHTTPRequestHandler):
    """
    Endpoints:
      GET /token?room=<room>&identity=<id>  → { "token": "...", "url": "..." }
      GET /health                            → { "status": "ok" }
    """

    def log_message(self, fmt, *args):
        logger.info(fmt % args)

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")   # allow browser requests
        self.end_headers()
        self.wfile.write(body)

    def _send_cors_preflight(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self._send_cors_preflight()

    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if parsed.path == "/health":
            self._send_json({"status": "ok", "api_key": LIVEKIT_API_KEY})

        elif parsed.path == "/token":
            room     = params.get("room",     ["bank-support"])[0]
            identity = params.get("identity", [f"user-{int(time.time())}"])[0]
            role     = params.get("role",     ["user"])[0]   # "user" or "agent"

            can_publish   = True
            can_subscribe = True

            token = create_livekit_token(
                api_key=LIVEKIT_API_KEY,
                api_secret=LIVEKIT_API_SECRET,
                room=room,
                identity=identity,
                can_publish=can_publish,
                can_subscribe=can_subscribe,
            )

            from config import LIVEKIT_URL
            self._send_json({
                "token":    token,
                "url":      LIVEKIT_URL,
                "room":     room,
                "identity": identity,
            })
            logger.info(f"Token issued → room={room}, identity={identity}")

        else:
            self._send_json({"error": "not found"}, status=404)


# ── Entry Point ────────────────────────────────────────────────────────────────

def run(port: int = 8080):
    server = HTTPServer(("0.0.0.0", port), TokenHandler)
    logger.info(f"Token server running on http://localhost:{port}")
    logger.info(f"  GET http://localhost:{port}/token?room=bank-support&identity=test-user")
    logger.info(f"  GET http://localhost:{port}/health")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Token server stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LiveKit Token Server")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    run(port=args.port)
