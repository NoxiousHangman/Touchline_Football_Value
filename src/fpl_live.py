"""Free Premier League feed and experimental in-season GBP estimates.

Refresh: python -m src.fpl_live
No account, API key, subscription, FPL prices or fantasy assists are used.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import sklearn

from .api_client import atomic_write
from .config import MODELS_DIR, RAW_DIR, PROCESSED_DIR
from .features import numbers
from .inseason import NUMERIC, FEATURES, MIN_MINUTES
from .market_history import add_market_history
from .player_identity import normalise, match_profiles
from .squad_catalog import fpl_squad, expand_squad

BASE_URL = "https://fantasy.premierleague.com/api"
LIVE_PATH = PROCESSED_DIR / "inseason-live.json"
MODEL_PATH = MODELS_DIR / "inseason-fpl-v1/model.joblib"


def make_live_features(bootstrap, fixtures, players, valuations, now, competition_code="GB1", league="Premier League"):
    """Only exact, unambiguous full-name and birthday matches get estimates."""
    events = bootstrap.get("events", [])
    if not events or not bootstrap.get("elements"):
        raise ValueError("FPL returned no season or player data.")
    season = pd.Timestamp(events[0]["deadline_time"]).year
    active_season = now.year if now.month >= 7 else now.year - 1
    if season != active_season:
        raise ValueError("FPL data does not belong to the current season.")
    if any(game.get("started") and not game.get("finished") for game in fixtures):
        raise ValueError("A match is in progress; retain the last completed-match snapshot.")
    finished = [game for game in fixtures if game.get("finished") and game.get("kickoff_time")]
    if not finished:
        raise ValueError("No completed Premier League fixtures yet.")
    latest = max(pd.Timestamp(game["kickoff_time"]) for game in finished)
    if latest > pd.Timestamp(now) or (pd.Timestamp(now) - latest).days > 21:
        raise ValueError("Completed fixture dates are future-dated or stale.")
    # Cumulative bootstrap totals have a fetch-time as-of, not an invented match date.
    as_of = now.date().isoformat()
    days = (now.date() - datetime(season, 7, 1).date()).days
    if not 60 <= days <= 300:
        raise ValueError("This experimental model supports September through April only.")
    fpl = pd.DataFrame(bootstrap["elements"])
    required = ["id", "first_name", "second_name", "birth_date", "team", "minutes", "goals_scored", "yellow_cards", "red_cards"]
    if not set(required).issubset(fpl.columns):
        raise ValueError("FPL player schema changed; refresh stopped.")
    records = [dict(id=str(a["id"]), name=a["first_name"] + " " + a["second_name"],
                    dob=a.get("birth_date"), aliases=[a.get("web_name"), a.get("known_name"), *a.get("aliases", [])])
               for a in bootstrap["elements"]]
    matches = match_profiles(records, players)
    fpl = fpl[required].copy()
    fpl["player_id"] = fpl.id.astype(str).map(matches)
    fpl = fpl.loc[fpl.player_id.notna()].copy()
    fpl["player_id"] = fpl.player_id.astype("int64")
    joined = fpl.merge(players[["player_id", "name", "date_of_birth", "position"]], on="player_id", validate="one_to_one")
    joined["dob"] = joined.date_of_birth.astype(str).str[:10]
    mapping = joined[["id", "player_id", "name", "dob"]].rename(columns={"id": "fpl_id", "name": "player_name"})
    mapping["match_basis"] = "unique exact published name alias and exact date of birth"
    joined["minutes"] = numbers(joined.minutes, "FPL minutes", nullable=False, integer=True)
    frame = joined.loc[joined.minutes >= MIN_MINUTES].copy()
    frame = frame.rename(columns={"name": "player_name", "minutes": "minutes_played", "goals_scored": "goals"})
    frame["age"] = (pd.Timestamp(as_of) - pd.to_datetime(frame.dob, errors="coerce")).dt.days / 365.2425
    frame = frame.loc[frame.age.between(14, 60) & frame.position.notna()].copy()
    frame["window_start"] = f"{season}-07-01"
    frame["observation_date"] = as_of
    frame["season_start_year"] = season
    frame["season_days"] = days
    frame["competition_code"] = competition_code
    frame["league"] = league
    frame["player_id_namespace"] = "transfermarkt"
    frame["is_synthetic"] = False
    teams = {team["id"]: team["name"] for team in bootstrap.get("teams", [])}
    frame["club"] = frame.team.map(teams)
    for name in ("goals", "yellow_cards", "red_cards"):
        frame[name] = numbers(frame[name], f"FPL {name}", nullable=False, integer=True)
        frame[f"{name}_per90_shrunk"] = frame[name] * 90 / (frame.minutes_played + 450)
    if frame.empty:
        raise ValueError("No confidently matched players have enough minutes.")
    frame = add_market_history(frame, valuations)
    # No silent median substitution for missing preseason value in this live feed.
    missing_prior = int(frame.preseason_value_missing.sum())
    frame = frame.loc[frame.preseason_value_missing == 0].copy()
    if frame.empty:
        raise ValueError("No matched players have a qualifying preseason market value.")
    return frame, mapping, {"season": season, "feed_players": len(bootstrap["elements"]), "matched_players": len(mapping),
                             "unmatched_or_ambiguous_players": len(bootstrap["elements"]) - len(mapping),
                             "eligible_estimates": len(frame), "excluded_missing_preseason_value": missing_prior,
                             "latest_completed_fixture": latest.isoformat()}


def refresh_live(now=None):
    now = now or datetime.now(timezone.utc)
    artifact = joblib.load(MODEL_PATH)  # Trusted local artifact only.
    if artifact.get("kind") != "inseason-v1" or artifact["metadata"]["numeric_features"] != NUMERIC:
        raise ValueError("In-season model schema differs; refresh stopped.")
    if artifact["metadata"]["versions"]["sklearn"] != sklearn.__version__:
        raise ValueError("In-season model needs its recorded scikit-learn version.")
    payloads = {}
    for endpoint in ("bootstrap-static", "fixtures"):
        try:
            response = requests.get(f"{BASE_URL}/{endpoint}/", timeout=(10, 30))
            response.raise_for_status()
            payloads[endpoint] = response.json()
        except (requests.RequestException, ValueError):
            raise ValueError("The free Premier League feed is unavailable; keeping the previous snapshot.") from None
    players = pd.read_csv(RAW_DIR / "transfermarkt/players.csv.gz")
    values = pd.read_csv(RAW_DIR / "transfermarkt/player_valuations.csv.gz")
    frame, mapping, coverage = make_live_features(payloads["bootstrap-static"], payloads["fixtures"], players, values, now)
    rate = artifact["metadata"]["fx"]["eur_to_gbp"]
    frame["predicted_market_value_gbp"] = np.maximum(0, artifact["pipeline"].predict(frame[FEATURES])) * rate
    frame["preseason_value_gbp"] = frame.preseason_value_million_eur * 1_000_000 * rate
    frame["appearances"] = np.nan  # Not supplied by bootstrap; starts are not appearances.
    frame["assists"] = np.nan  # FPL fantasy assists have a different definition; do not mix sources.
    frame["goals_per90"] = frame.goals * 90 / frame.minutes_played
    frame["assists_per90"] = np.nan
    frame["predates_training_cutoff"] = False
    frame["missing_numeric_features"] = frame[NUMERIC].isna().sum(axis=1)
    frame["estimate_available"] = True
    fields = ["player_id", "player_name", "observation_date", "season_start_year", "position", "competition_code", "league", "club",
              "age", "minutes_played", "goals", "assists", "appearances", "goals_per90", "assists_per90", "predicted_market_value_gbp",
              "preseason_value_gbp", "preseason_valuation_date", "preseason_value_missing", "predates_training_cutoff",
              "missing_numeric_features", "estimate_available"]
    snapshot_id = now.strftime("%Y%m%dT%H%M%S%fZ")
    archive = RAW_DIR / "fpl" / snapshot_id
    archive.mkdir(parents=True)
    hashes = {}
    for name, payload in payloads.items():
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        (archive / f"{name}.json").write_bytes(encoded)
        hashes[name] = hashlib.sha256(encoded).hexdigest()
    mapping.to_csv(archive / "player_mapping.csv", index=False)
    result = {"status": "experimental", "fetched_at": now.isoformat(), "source": BASE_URL,
              "model_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(), "source_sha256": hashes,
              "coverage": coverage, "fx": artifact["metadata"]["fx"],
              "historical_metrics": artifact["metadata"]["metrics"],
              "historical_elite_metrics": artifact["metadata"]["value_bands"]["elite_50m_plus"],
              "limitations": ["This source covers Premier League; the other four leagues use ESPN.",
                              "Experimental cross-source estimates; not validated against current 2026/27 values.",
                              "Assists and appearance counts are unavailable in this feed integration.",
                              "Uses dated preseason Transfermarkt values, never fantasy game prices.",
                              "Model estimates require 180+ minutes and an exact published-name/birthday match. Other squad members have dated references or no price."],
              "players": json.loads(frame[fields].to_json(orient="records"))}
    result = expand_squad(result, fpl_squad(payloads["bootstrap-static"]), players, values,
                          "GB1", "Premier League", "FPL", now)
    atomic_write(archive / "metadata.json", json.dumps({k: v for k, v in result.items() if k != "players"}, indent=2) + "\n")
    atomic_write(LIVE_PATH, json.dumps(result, ensure_ascii=False, allow_nan=False))
    return result


if __name__ == "__main__":
    report = refresh_live()
    print(json.dumps({key: report[key] for key in ("status", "fetched_at", "coverage")}, indent=2))
