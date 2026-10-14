"""Shared, allowlisted HTTP routes for local and hosted Touchline servers."""
from http import HTTPStatus
import json
import logging
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from xml.sax.saxutils import escape

from .fee_adjustment import adjustment_info
from .player_filters import parse_filters

ROOT = Path(__file__).resolve().parents[1]
SECURITY_HEADERS = [
    ("X-Content-Type-Options", "nosniff"),
    ("X-Frame-Options", "DENY"),
    ("Referrer-Policy", "strict-origin-when-cross-origin"),
    ("Permissions-Policy", "camera=(), microphone=(), geolocation=()"),
    ("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; "
     "style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self'; "
     "frame-ancestors 'none'; base-uri 'none'; form-action 'self'; object-src 'none'"),
]


class WebApplication:
    def __init__(self, predictor, *, public=False, public_url=""):
        self.predictor = predictor
        self.public = public
        self.public_url = public_url.rstrip("/")
        if self.public_url:
            parts = urlsplit(self.public_url)
            if (parts.scheme != "https" or not parts.hostname or parts.username or
                    parts.password or parts.path or parts.query or parts.fragment):
                raise ValueError("PUBLIC_BASE_URL must be a bare HTTPS origin, e.g. https://touchline.onrender.com")

    def response(self, body, content_type="application/json; charset=utf-8", status=200):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, allow_nan=False, ensure_ascii=False).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        headers = [("Content-Type", content_type), ("Content-Length", str(len(body))),
                   ("Cache-Control", "no-store"), *SECURITY_HEADERS]
        if self.public and self.public_url:
            headers.append(("Strict-Transport-Security", "max-age=31536000"))
        return status, headers, body

    def dispatch(self, method, target, headers=None, *, local_port=8765):
        headers = {key.lower(): value for key, value in (headers or {}).items()}
        try:
            if len(target) > 2048:
                return self.response({"error": "Request URL is too long"}, status=414)
            url = urlsplit(target)
            if method == "POST" and url.path == "/api/refresh":
                # Visitors cannot force costly provider downloads on a public service.
                if self.public:
                    return self.response({"error": "Updates are managed automatically on this hosted site"}, status=403)
                origins = {f"http://127.0.0.1:{local_port}", f"http://localhost:{local_port}"}
                if (headers.get("x-touchline-refresh") != "1" or
                        headers.get("origin") not in origins | {None} or
                        headers.get("sec-fetch-site") == "cross-site"):
                    return self.response({"error": "Refresh must be requested from this app"}, status=403)
                payload = self.predictor.live.request_refresh(force=True)
                return self.response(payload, status=202 if payload["accepted"] else 200)
            if method not in ("GET", "HEAD"):
                return self.response({"error": "Method not allowed"}, status=405)
            if url.path == "/healthz":
                return self.response({"status": "ok"})
            pages = {"/": "index.html", "/live": "index.html", "/privacy": "privacy.html",
                     "/about": "about.html", "/favicon.svg": "favicon.svg"}
            if url.path in pages:
                kind = "image/svg+xml" if url.path == "/favicon.svg" else "text/html; charset=utf-8"
                return self.response((ROOT / "web" / pages[url.path]).read_bytes(), kind)
            if url.path == "/robots.txt":
                sitemap = f"Sitemap: {self.public_url}/sitemap.xml\n" if self.public_url else ""
                return self.response("User-agent: *\nAllow: /\nDisallow: /api/\n" + sitemap, "text/plain; charset=utf-8")
            if url.path == "/sitemap.xml" and self.public_url:
                entries = "".join(f"<url><loc>{escape(self.public_url + path)}</loc></url>"
                                  for path in ("/live", "/", "/about", "/privacy"))
                return self.response('<?xml version="1.0" encoding="UTF-8"?>'
                                     '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
                                     + entries + '</urlset>', "application/xml; charset=utf-8")
            query = parse_qs(url.query, max_num_fields=20)
            live = query.get("mode", [""])[0] == "live"
            if url.path == "/api/players":
                offset = max(0, int(query.get("offset", ["0"])[0]))
                limit = max(1, min(100, int(query.get("limit", ["40"])[0])))
                payload = self.predictor.search(query.get("q", [""])[0][:200], query.get("league", [""])[0],
                                                live, offset, limit, parse_filters(query, live))
            elif url.path.startswith("/api/player/"):
                try:
                    player_id = int(url.path.rsplit("/", 1)[1])
                except ValueError:
                    return self.response({"error": "Player not found"}, status=404)
                payload = self.predictor.player(player_id, live)
                if payload is None:
                    return self.response({"error": "Player not found"}, status=404)
            elif url.path == "/api/info":
                if live:
                    data = self.predictor.live.data()
                    payload = {key: data.get(key) for key in (
                        "available", "message", "fetched_at", "leagues", "revision", "model_estimates",
                        "preseason_references", "profile_only")}
                    payload.update(players=len(data["players"]), fee_adjustment=adjustment_info(),
                                   update=self.predictor.live.refresh_status(), manual_refresh=not self.public,
                                   hosted=self.public,
                                   mae_gbp=data.get("historical_metrics", {}).get("mae_gbp"),
                                   elite_mae_gbp=data.get("historical_elite_metrics", {}).get("mae_gbp"))
                else:
                    meta = self.predictor.artifact["metadata"]
                    payload = {"players": len(self.predictor.latest), "model": "Preseason value + performance",
                               "fx": meta["fx"], "mae_gbp": meta["metrics"]["mae_gbp"],
                               "elite_mae_gbp": meta["value_band_metrics"]["elite_50m_plus"]["mae_gbp"]}
            else:
                return self.response({"error": "Not found"}, status=404)
            return self.response(payload)
        except ValueError:
            return self.response({"error": "Invalid request parameters"}, status=400)
        except Exception:
            logging.exception("Request failed: %s", urlsplit(target).path)
            return self.response({"error": "The app could not complete this request. Please try again."}, status=500)

    def __call__(self, environ, start_response):
        target = environ.get("PATH_INFO", "/")
        if environ.get("QUERY_STRING"):
            target += "?" + environ["QUERY_STRING"]
        headers = {key[5:].replace("_", "-"): value for key, value in environ.items() if key.startswith("HTTP_")}
        method = environ.get("REQUEST_METHOD", "GET")
        status, response_headers, body = self.dispatch(method, target, headers)
        start_response(f"{status} {HTTPStatus(status).phrase}", response_headers)
        return [b"" if method == "HEAD" else body]
