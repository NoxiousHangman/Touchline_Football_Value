"""Small synchronous football-data.org v4 client with a local JSON cache."""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any

import requests

from .config import API_BASE_URL, ATTRIBUTION, RAW_DIR, TOKEN_ENV_VAR


class FootballDataError(Exception):
    """A safe-to-display configuration, API, or cache error."""


@dataclass(frozen=True)
class FetchResult:
    data: dict[str, Any]
    cache_path: Path
    from_cache: bool
    fetched_at: str


def atomic_write(path: Path, text: str) -> None:
    """Replace a file only after its complete contents have been written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent,
            suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class FootballDataClient:
    """Read the token only from the environment; never store it in cache metadata.

    Cache hits need no token. Offline mode accepts old cache entries explicitly.
    Live requests are spaced 6.1 seconds apart within this client instance.
    """

    def __init__(self, cache_dir: Path = RAW_DIR, cache_ttl: float = 86400):
        if not math.isfinite(cache_ttl) or cache_ttl < 0:
            raise ValueError("cache_ttl must be a finite non-negative number")
        self.cache_dir = Path(cache_dir)
        self.cache_ttl = cache_ttl
        self._session = requests.Session()
        self._last_request: float | None = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._session.close()

    def fetch(self, endpoint: str, params: dict | None = None, *,
              refresh: bool = False, offline: bool = False) -> FetchResult:
        if refresh and offline:
            raise ValueError("refresh and offline cannot be combined")
        # Restrict requests to the provider and known resource paths.
        if not re.fullmatch(
            r"/(?:competitions/[A-Za-z0-9]+/(?:scorers|matches)|persons/[0-9]+(?:/matches)?)",
            endpoint,
        ):
            raise ValueError("Unsupported endpoint")
        params = {key: str(value) for key, value in (params or {}).items() if value is not None}
        request = {"url": API_BASE_URL + endpoint, "params": params}
        key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:20]
        path = self.cache_dir / f"{endpoint.strip('/').replace('/', '_')}_{key}.json"
        if not refresh and path.exists():
            try:
                cached = json.loads(path.read_text(encoding="utf-8"))
                fetched_at = cached["fetched_at"]
                timestamp = datetime.fromisoformat(fetched_at)
                if (cached["request"] != request or not isinstance(cached["data"], dict)
                        or timestamp.tzinfo is None):
                    raise ValueError("Invalid cache structure")
                age = (datetime.now(timezone.utc) - timestamp).total_seconds()
                if offline or 0 <= age < self.cache_ttl:
                    return FetchResult(cached["data"], path, True, fetched_at)
            except (ValueError, KeyError, TypeError):
                if offline:
                    raise FootballDataError("Cached response is invalid; rerun online with --refresh.") from None
                # A malformed cache entry can be safely replaced by a live response.
        if offline:
            raise FootballDataError("No matching cached response. Fetch this request online first.")

        token = os.environ.get(TOKEN_ENV_VAR, "").strip()
        if not token:
            raise FootballDataError(f"Set {TOKEN_ENV_VAR} before fetching live data. See README.md.")
        if any(char.isspace() for char in token):
            raise FootballDataError(f"{TOKEN_ENV_VAR} must not contain whitespace.")
        if self._last_request is not None:
            time.sleep(max(0, 6.1 - (time.monotonic() - self._last_request)))
        self._last_request = time.monotonic()
        try:
            response = self._session.get(
                request["url"], params=params,
                headers={"X-Auth-Token": token, "Accept": "application/json"},
                timeout=(10, 30), allow_redirects=False,
            )
        except requests.Timeout:
            raise FootballDataError("API request timed out. Try again later.") from None
        except requests.RequestException:
            # Do not echo request objects, headers, or exception text containing secrets.
            raise FootballDataError("Could not connect to football-data.org. Check your connection.") from None
        with response:
            status = response.status_code
            if status != 200:
                messages = {
                    400: "Invalid request/filter. Check the supplied IDs, season and dates.",
                    401: "Authentication failed. Check your API token.",
                    403: "Access denied. Check your token and subscription coverage for this resource/season.",
                    404: "Resource not found. Check the competition or player ID.",
                    429: "API rate limit reached. Wait before retrying.",
                }
                message = messages.get(status, "Provider unavailable or unexpected response. Try again later.")
                if status == 429:
                    delay = response.headers.get("Retry-After") or response.headers.get("X-RequestCounter-Reset")
                    if delay and delay.isdigit():
                        message += f" Retry after {delay} seconds."
                raise FootballDataError(f"HTTP {status}: {message}")
            try:
                data = response.json()
            except ValueError:
                raise FootballDataError("API returned invalid JSON; nothing was cached.") from None
        if not isinstance(data, dict) or "errorCode" in data:
            raise FootballDataError("API returned an unexpected payload; nothing was cached.")
        fetched_at = datetime.now(timezone.utc).isoformat()
        envelope = {"source": "football-data.org", "attribution": ATTRIBUTION,
                    "fetched_at": fetched_at, "request": request, "data": data}
        atomic_write(path, json.dumps(envelope, indent=2, ensure_ascii=False) + "\n")
        return FetchResult(data, path, False, fetched_at)

    def competition_scorers(self, competition: str = "PL", *, season: int | None = None,
                            limit: int = 10, **options) -> FetchResult:
        return self.fetch(f"/competitions/{competition}/scorers",
                          {"season": season, "limit": limit}, **options)

    def competition_matches(self, competition: str = "PL", *, season: int | None = None,
                            **options) -> FetchResult:
        return self.fetch(f"/competitions/{competition}/matches", {"season": season}, **options)

    def player(self, player_id: int, **options) -> FetchResult:
        return self.fetch(f"/persons/{player_id}", **options)

    def player_matches(self, player_id: int, *, date_from: str | None = None,
                       date_to: str | None = None, competitions: str | None = None,
                       limit: int = 100, offset: int | None = None, **options) -> FetchResult:
        """Fetch one explicit window; provider aggregates describe only returned matches."""
        return self.fetch(f"/persons/{player_id}/matches", {
            "dateFrom": date_from, "dateTo": date_to, "competitions": competitions,
            "limit": limit, "offset": offset,
        }, **options)
