"""Local, read-only interface to the trained predictor. Run: python app.py"""

import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import math
import os
import gzip
from pathlib import Path
import unicodedata
import threading
import time
from urllib.parse import parse_qs, urlsplit

import pandas as pd

from src.features import build_features
from src.predict import load_model, predict_players
from src.fee_adjustment import adjustment_info, with_fee_scenario
from src.player_filters import filter_players, parse_filters

ROOT = Path(__file__).resolve().parent
LEAGUES = {"GB1": "Premier League", "ES1": "La Liga", "L1": "Bundesliga", "IT1": "Serie A", "FR1": "Ligue 1"}


class LiveStore:
    """Refresh independently of browser traffic while the local server runs."""
    def __init__(self):
        self.lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.data_lock = threading.Lock()
        self.cached_data = None
        self.cached_at = float('-inf')
        self.last_attempt = float('-inf')
        self.stop_event = threading.Event()
        self.updater = None
        self.update_state = {"last_checked_at": None, "started_at": None,
                             "finished_at": None, "current_league": None, "leagues": {}}

    def start_updater(self):
        if self.updater and self.updater.is_alive():
            return
        self.stop_event.clear()
        self.updater = threading.Thread(target=self._watch, name="live-update-checker", daemon=True)
        self.updater.start()

    def stop_updater(self):
        self.stop_event.set()
        if self.updater:
            self.updater.join(timeout=1)

    def _watch(self):
        while not self.stop_event.is_set():
            try:
                self.request_refresh()
            except Exception:
                logging.exception("Automatic update check failed; it will be retried")
            self.stop_event.wait(60)

    def refresh_status(self):
        with self.state_lock:
            state = dict(self.update_state, leagues={k: dict(v) for k, v in self.update_state["leagues"].items()})
        state.update(running=self.lock.locked(),
                     automatic=bool(self.updater and self.updater.is_alive() and not self.stop_event.is_set()),
                     retry_after_seconds=math.ceil(max(0, 900 - (time.monotonic() - self.last_attempt))))
        return state

    def request_refresh(self, force=False):
        with self.state_lock:
            self.update_state["last_checked_at"] = datetime.now(timezone.utc).isoformat()
        if not self.lock.acquire(blocking=False):
            return {"accepted": False, "reason": "running", "update": self.refresh_status()}
        try:
            fresh = not force and all(self.age(self.read_cache(p)) <= 6 * 3600 for p in self.paths().values())
        except Exception:
            self.lock.release()
            raise
        if fresh:
            self.lock.release()
            return {"accepted": False, "reason": "fresh", "update": self.refresh_status()}
        if time.monotonic() - self.last_attempt < 900:
            self.lock.release()
            return {"accepted": False, "reason": "cooldown", "update": self.refresh_status()}
        self.last_attempt = time.monotonic()
        with self.state_lock:
            self.update_state.update(started_at=datetime.now(timezone.utc).isoformat(), finished_at=None)
        try:
            threading.Thread(target=self._refresh, kwargs={"force": force}, name="live-data-refresh", daemon=True).start()
        except Exception:
            self.lock.release()
            raise
        return {"accepted": True, "reason": "started", "update": self.refresh_status()}

    def _refresh(self, force=False):
        try:
            from src.fpl_live import refresh_live
            from src.espn_live import refresh_league
            for code, path in self.paths().items():
                if not force and self.age(self.read_cache(path)) <= 6 * 3600:
                    continue
                with self.state_lock:
                    self.update_state["current_league"] = LEAGUES[code]
                    previous = self.update_state["leagues"].get(code, {})
                    self.update_state["leagues"][code] = dict(previous, checking=True,
                        attempted_at=datetime.now(timezone.utc).isoformat(), error=None)
                error = None
                try:
                    if code == "GB1":
                        refresh_live()
                    elif force:
                        refresh_league(code, force=True)
                    else:
                        refresh_league(code)
                except Exception as exc:
                    # Preserve the last atomic snapshot and continue with other leagues.
                    logging.exception("%s update failed", LEAGUES[code])
                    error = ("The source did not provide verifiable current-season data." if isinstance(exc, (ValueError, KeyError))
                             else "The source could not be updated. The previous snapshot was kept.")
                with self.state_lock:
                    state = self.update_state["leagues"][code]
                    state.update(checking=False, error=error)
                    if not error:
                        state["last_success_at"] = datetime.now(timezone.utc).isoformat()
        finally:
            with self.state_lock:
                self.update_state.update(finished_at=datetime.now(timezone.utc).isoformat(), current_league=None)
            self.lock.release()

    @staticmethod
    def paths():
        from src.fpl_live import LIVE_PATH
        from src.espn_live import LEAGUES, cache_path
        return {"GB1": LIVE_PATH, **{code: cache_path(code) for code in LEAGUES}}

    @staticmethod
    def read_cache(path):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(result, dict) or not isinstance(result.get("players", []), list) or not isinstance(result.get("coverage", {}), dict):
                return None
            return result
        except (OSError, ValueError):
            return None

    @staticmethod
    def age(result):
        try:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(result["fetched_at"])).total_seconds()
            season = datetime.now(timezone.utc).year - (datetime.now(timezone.utc).month < 7)
            if result.get("coverage", {}).get("season", season) != season or age < 0:
                return float("inf")
            return age
        except (TypeError, KeyError, ValueError):
            return float("inf")

    def data(self):
        # Bound parsing and Pandas work under concurrent public requests. A short
        # cache also prevents four HTTP threads constructing the same large view.
        with self.data_lock:
            if self.cached_data is None or time.monotonic() - self.cached_at >= 10:
                self.cached_data = self._data()
                self.cached_at = time.monotonic()
            return self.cached_data

    def _data(self):
        cached = {code: self.read_cache(path) for code, path in self.paths().items()}
        if os.environ.get("TOUCHLINE_AUTO_UPDATE", "1") != "0" and any(self.age(r) > 6 * 3600 for r in cached.values()):
            self.request_refresh()
        result = {"players": [], "available": False, "leagues": {}}
        for code, snapshot in cached.items():
            available = self.age(snapshot) <= 48 * 3600
            rows = snapshot.get("players", []) if available else []
            result["leagues"][code] = {"name": LEAGUES[code], "available": available, "players": len(rows),
                                      "fetched_at": snapshot.get("fetched_at") if snapshot else None,
                                      "refresh_due": self.age(snapshot) > 6 * 3600,
                                      "club_warnings": snapshot.get("coverage", {}).get("club_warnings", []) if available else [],
                                      "latest_completed_fixture": snapshot.get("coverage", {}).get("latest_completed_fixture") if snapshot else None}
            if not available:
                continue
            for row in rows:
                result["players"].append(dict(row, fetched_at=snapshot["fetched_at"],
                    statistics_source="FPL" if code == "GB1" else "ESPN"))
            for field in ("historical_metrics", "historical_elite_metrics"):
                result[field] = snapshot.get(field, {})
        # A unique eligible model row takes precedence over a duplicate roster-only
        # listing (common around transfers). Otherwise retain the ambiguity guard.
        counts = pd.Series([r["player_id"] for r in result["players"]], dtype="int64").value_counts()
        model_counts = pd.Series([r["player_id"] for r in result["players"] if r.get("estimate_available")], dtype="int64").value_counts()
        result["players"] = [r for r in result["players"] if counts[r["player_id"]] == 1 or
                             (r.get("estimate_available") and model_counts.get(r["player_id"], 0) == 1)]
        for code, info in result["leagues"].items():
            info["players"] = sum(r["competition_code"] == code for r in result["players"])
        result["available"] = bool(result["players"])
        result["model_estimates"] = sum(bool(r.get("estimate_available")) for r in result["players"])
        result["preseason_references"] = sum(r.get("valuation_basis") == "preseason_reference" for r in result["players"])
        result["profile_only"] = sum(r.get("valuation_basis") == "profile_only" for r in result["players"])
        dates = [v["fetched_at"] for v in result["leagues"].values() if v["available"]]
        result["fetched_at"] = min(dates) if dates else None
        result["revision"] = "|".join(f"{k}:{v['fetched_at']}:{v['available']}" for k, v in result["leagues"].items())
        if not result["available"]:
            result["message"] = "The current-season feeds need a refresh. Please try again shortly."
        return result


def search_key(value):
    return "".join(c for c in unicodedata.normalize("NFKD", str(value).casefold()) if not unicodedata.combining(c))


class Predictor:
    def __init__(self):
        self.live = LiveStore()
        snapshot_path = os.environ.get("TOUCHLINE_HISTORY_PATH")
        if snapshot_path:
            with gzip.open(snapshot_path, "rt", encoding="utf-8") as handle:
                snapshot = json.load(handle)
            if snapshot.get("schema_version") != 1:
                raise ValueError("Unsupported deployment history snapshot; rebuild the release.")
            self.artifact = {"metadata": snapshot["metadata"]}
            self.rows = pd.DataFrame(snapshot["rows"])
        else:
            self.artifact = load_model(ROOT / "models/transfermarkt-elite/model.joblib")
            features = build_features(pd.read_csv(ROOT / "data/processed/transfermarkt-real/features.csv"))
            history = pd.read_csv(ROOT / "data/raw/transfermarkt/player_valuations.csv.gz")
            predicted = predict_players(features, self.artifact, valuation_history=history)
            fields = ["player_id", "observation_date", "season_start_year", "position", "competition_code", "age",
                      "minutes_played", "goals", "assists", "appearances", "goals_per90", "assists_per90"]
            self.rows = predicted.merge(features[fields], on=["player_id", "observation_date"], validate="one_to_one")
            # The labelled training population requires 450 minutes. Per-90 outliers
            # outside that population should not be presented as usable estimates.
            self.rows.loc[self.rows.minutes_played < 450, "predicted_market_value_gbp"] = float("nan")
        self.latest = self.rows.sort_values("observation_date").drop_duplicates("player_id", keep="last").copy()
        self.latest["search_key"] = self.latest.player_name.map(search_key)

    @staticmethod
    def record(row):
        # Round-trip through Pandas to turn numpy values/NaN into JSON numbers/null.
        names = ["player_id", "player_name", "observation_date", "season_start_year", "position", "competition_code",
                 "age", "minutes_played", "goals", "assists", "appearances", "goals_per90", "assists_per90",
                 "predicted_market_value_gbp", "preseason_value_gbp", "preseason_valuation_date",
                 "preseason_value_missing", "predates_training_cutoff", "missing_numeric_features"]
        result = json.loads(row[names].to_json())
        result["estimate_available"] = result["minutes_played"] >= 450
        if not result["estimate_available"]:
            result["predicted_market_value_gbp"] = None
        result["league"] = LEAGUES.get(result["competition_code"], result["competition_code"])
        return result

    def search(self, query="", league="", live=False, offset=0, limit=40, filters=None):
        if live:
            data = self.live.data()
            rows = [p for p in data["players"] if (not league or p["competition_code"] == league) and
                    all(token in search_key(p["player_name"] + " " + str(p.get("club", ""))) for token in search_key(query).split())]
            rows = filter_players(rows, filters or parse_filters({}, live=True))
            end = offset + limit
            return {"total": len(rows), "offset": offset, "next_offset": end if end < len(rows) else None,
                    "players": [with_fee_scenario(row) for row in rows[offset:end]]}
        rows = self.latest
        if query.strip():
            tokens = search_key(query).split()
            for token in tokens:
                rows = rows.loc[rows.search_key.str.contains(token, regex=False)]
        if league:
            rows = rows.loc[rows.competition_code == league]
        rows = rows.sort_values(["predicted_market_value_gbp", "player_name"], ascending=[False, True])
        end = offset + limit
        if filters:
            records = filter_players([self.record(row) for _, row in rows.iterrows()], filters)
            return {"total": len(records), "offset": offset, "next_offset": end if end < len(records) else None,
                    "players": records[offset:end]}
        return {"total": len(rows), "offset": offset, "next_offset": end if end < len(rows) else None,
                "players": [self.record(row) for _, row in rows.iloc[offset:end].iterrows()]}

    def player(self, player_id, live=False):
        if live:
            rows = [p for p in self.live.data()["players"] if p["player_id"] == player_id]
            return {"seasons": [with_fee_scenario(row) for row in rows]} if rows else None
        rows = self.rows.loc[self.rows.player_id == player_id].sort_values("observation_date", ascending=False)
        if rows.empty:
            return None
        return {"seasons": [self.record(row) for _, row in rows.iterrows()]}


def handler_for(predictor):
    # Local development and the hosted WSGI service share exactly the same routes.
    from src.web_server import WebApplication
    service = WebApplication(predictor)

    class Handler(BaseHTTPRequestHandler):
        def dispatch(self):
            status, headers, body = service.dispatch(
                self.command, self.path, dict(self.headers),
                local_port=self.server.server_address[1])
            self.send_response(status)
            for name, value in headers:
                self.send_header(name, value)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        do_GET = dispatch
        do_HEAD = dispatch
        do_POST = dispatch

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    print("Loading the trained predictor...", flush=True)
    predictor = Predictor()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(predictor))
    predictor.live.start_updater()
    print(f"Ready: http://127.0.0.1:{args.port} ({len(predictor.latest)} players)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        predictor.live.stop_updater()
        server.server_close()


if __name__ == "__main__":
    main()
