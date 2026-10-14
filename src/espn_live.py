"""Free, keyless ESPN season statistics for the other four major leagues.

Only public endpoints are used. ESPN is not an officially supported API here.
No paid services, scraped credentials, or fantasy prices are involved.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import time

import joblib
import numpy as np
import pandas as pd
import requests
import sklearn

from .api_client import atomic_write
from .config import RAW_DIR, PROCESSED_DIR
from .fpl_live import MODEL_PATH, make_live_features, normalise
from .inseason import FEATURES, NUMERIC
from .player_identity import match_profiles
from .squad_catalog import espn_squad, expand_squad

LEAGUES = {"ES1": ("esp.1", "La Liga", 20), "L1": ("ger.1", "Bundesliga", 18),
           "IT1": ("ita.1", "Serie A", 20), "FR1": ("fra.1", "Ligue 1", 18)}
SITE = "https://site.api.espn.com/apis/site/v2/sports/soccer"
CORE = "https://sports.core.api.espn.com/v2/sports/soccer/leagues"


def cache_path(code):
    return PROCESSED_DIR / f"inseason-live-{code}.json"


class PublicClient:
    """Archive responses and reuse a six-hour HTTP cache; two workers at most."""
    def __init__(self, archive, now, force=False):
        self.archive, self.now = archive, now
        self.manifest = []
        self.force = force

    def get(self, url):
        key = hashlib.sha256(url.encode()).hexdigest()
        path = RAW_DIR / "espn/http" / (key + ".json")
        cached, age = None, float("inf")
        if not self.force:
            try:
                cached = json.loads(path.read_text(encoding="utf8"))
                age = (self.now - datetime.fromisoformat(cached["fetched_at"])).total_seconds()
                cached["data"]
            except (OSError, ValueError, KeyError, TypeError):
                cached, age = None, float("inf")
        if self.force or not 0 <= age < 6 * 3600:
            time.sleep(0.1)
            response = requests.get(url, timeout=(10, 30))
            response.raise_for_status()
            cached = {"url": url, "fetched_at": datetime.now(timezone.utc).isoformat(), "data": response.json()}
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write(path, json.dumps(cached, ensure_ascii=False, allow_nan=False))
        encoded = json.dumps(cached, ensure_ascii=False, allow_nan=False)
        atomic_write(self.archive / (key + ".json"), encoded)
        self.manifest.append({"url": url, "file": key + ".json", "fetched_at": cached["fetched_at"],
                              "sha256": hashlib.sha256(encoded.encode()).hexdigest()})
        return cached["data"]


def validate_season(payload, season):
    if payload.get("season", {}).get("year") != season:
        raise ValueError("ESPN returned a different season; previous snapshot retained.")


def flatten_stats(payload):
    stats = [s for c in payload.get("splits", {}).get("categories", []) for s in c.get("stats", [])]
    names = [s["name"] for s in stats]
    if len(names) != len(set(names)):
        raise ValueError("Ambiguous ESPN statistics.")
    return {s["name"]: s["value"] for s in stats}


def read_statistics(payload, slug, season, athlete_id):
    expected = f"/soccer/leagues/{slug}/seasons/{season}/types/1/athletes/{athlete_id}/statistics"
    if expected not in payload.get("$ref", ""):
        raise ValueError("Wrong player, league or season in ESPN statistics.")
    stats = flatten_stats(payload)
    required = ("minutes", "totalGoals", "yellowCards", "redCards", "appearances", "goalAssists")
    if any(name not in stats for name in required):
        raise ValueError("ESPN omitted a required statistic; it cannot be assumed zero.")
    for name in required:
        value = stats[name]
        if not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0 or value != int(value):
            raise ValueError("Invalid ESPN count.")
    if stats["minutes"] > stats["appearances"] * 100:
        raise ValueError("ESPN minutes and appearances disagree.")
    return stats


def read_fixtures(payload, slug, season, now):
    validate_season(payload, season)
    fixtures = []
    for event in payload.get("events", []):
        if event.get("league", {}).get("slug") != slug:
            continue  # Cup and continental fixtures must not enter league totals.
        validate_season(event, season)
        for match in event.get("competitions", []):
            state = match.get("status", {}).get("type", {})
            date = pd.Timestamp(event["date"])
            if state.get("state") == "in":
                raise ValueError("An ESPN league match is in progress; previous snapshot retained.")
            if state.get("completed"):
                if date > pd.Timestamp(now) or date < pd.Timestamp(f"{season}-07-01", tz="UTC"):
                    raise ValueError("Invalid completed fixture date.")
                fixtures.append({"started": True, "finished": True, "kickoff_time": event["date"]})
    if not fixtures or (pd.Timestamp(now) - max(pd.Timestamp(f["kickoff_time"]) for f in fixtures)).days > 21:
        raise ValueError("No recent completed league fixtures in ESPN schedule.")
    return fixtures


def read_team_data(client, team, slug, season, now):
    """A bad club schedule must not suppress verified data for every other club."""
    roster = client.get(f"{SITE}/{slug}/teams/{team['id']}/roster?season={season}")
    validate_season(roster, season)
    if roster.get("team", {}).get("id") != team["id"]:
        raise ValueError("ESPN roster team differs.")
    warning = None
    try:
        schedule = client.get(f"{SITE}/{slug}/teams/{team['id']}/schedule?season={season}")
        fixtures = read_fixtures(schedule, slug, season, now)
    except (requests.RequestException, ValueError, KeyError):
        fixtures = []
        warning = "Current league fixtures could not be verified; match statistics and model estimates are unavailable for this club."
    athletes = [dict(a, club_id=team["id"], statistics_unavailable_reason=warning) for a in roster["athletes"]]
    return athletes, fixtures, warning


def refresh_league(code, now=None, force=False):
    now = now or datetime.now(timezone.utc)
    slug, name, team_count = LEAGUES[code]
    season = now.year if now.month >= 7 else now.year - 1
    archive = RAW_DIR / "espn" / code / now.strftime("%Y%m%dT%H%M%S%fZ")
    archive.mkdir(parents=True)
    client = PublicClient(archive, now, force=force)
    artifact = joblib.load(MODEL_PATH)
    meta = artifact["metadata"]
    if artifact.get("kind") != "inseason-v1" or meta["numeric_features"] != NUMERIC or meta["versions"]["sklearn"] != sklearn.__version__:
        raise ValueError("In-season model schema or library version differs.")
    players = pd.read_csv(RAW_DIR / "transfermarkt/players.csv.gz")
    values = pd.read_csv(RAW_DIR / "transfermarkt/player_valuations.csv.gz")
    teams_payload = client.get(f"{SITE}/{slug}/teams?limit=100")
    league = teams_payload["sports"][0]["leagues"][0]
    validate_season(league, season)
    if league.get("slug") != slug:
        raise ValueError("Wrong ESPN league.")
    teams = [row["team"] for row in league["teams"]]
    if len(teams) != team_count or len({t["id"] for t in teams}) != team_count:
        raise ValueError("Incomplete or duplicate ESPN team list.")
    print(f"{name}: checking {len(teams)} current squads...", flush=True)

    def team_data(team):
        return read_team_data(client, team, slug, season, now)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(team_data, teams))
    athletes = [a for rows, _, _ in results for a in rows]
    fixtures = [f for _, rows, _ in results for f in rows]
    warnings = [{"club": team["displayName"], "message": result[2]}
                for team, result in zip(teams, results) if result[2]]
    if not fixtures:
        raise ValueError("No verified current league fixtures; previous snapshot retained.")
    identity_rows = [dict(id=str(a["id"]), name=a["fullName"], dob=a.get("dateOfBirth"),
                          aliases=[a.get("displayName")]) for a in athletes]
    matched = match_profiles(identity_rows, players)
    candidates = [a for a in athletes if not a.get("statistics_unavailable_reason") and str(a["id"]) in matched and
                  flatten_stats(a.get("statistics", {})).get("appearances", 0) >= 2]
    print(f"{name}: reading season totals for {len(candidates)} matched players...", flush=True)
    omitted = []

    def athlete_data(a):
        url = f"{CORE}/{slug}/seasons/{season}/types/1/athletes/{a['id']}/statistics"
        try:
            stats = read_statistics(client.get(url), slug, season, a["id"])
            roster_stats = flatten_stats(a["statistics"])
            if any(stats[k] != roster_stats.get(k) for k in ("appearances", "totalGoals", "yellowCards", "redCards")):
                raise ValueError("Roster and detailed season totals disagree.")
            return dict(id=int(a["id"]), aliases=[a.get("displayName")], first_name=a["fullName"], second_name="", birth_date=a["dateOfBirth"][:10],
                        team=a["club_id"], minutes=stats["minutes"], goals_scored=stats["totalGoals"],
                        yellow_cards=stats["yellowCards"], red_cards=stats["redCards"],
                        appearances=stats["appearances"], assists=stats["goalAssists"])
        except (requests.RequestException, ValueError) as exc:
            omitted.append({"espn_id": a["id"], "reason": type(exc).__name__})
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        elements = [a for a in pool.map(athlete_data, candidates) if a is not None]
    bootstrap = {"events": [{"deadline_time": f"{season}-07-01T00:00:00Z"}], "elements": elements,
                 "teams": [{"id": t["id"], "name": t["displayName"]} for t in teams]}
    frame, mapping, coverage = make_live_features(bootstrap, fixtures, players, values, now, code, name)
    mapping = mapping.rename(columns={"fpl_id": "espn_id"})
    additional = pd.DataFrame(elements).set_index("id")
    frame["appearances"] = frame.id.map(additional.appearances)
    frame["assists"] = frame.id.map(additional.assists)
    frame["goals_per90"] = frame.goals * 90 / frame.minutes_played
    frame["assists_per90"] = frame.assists * 90 / frame.minutes_played
    frame["predicted_market_value_gbp"] = np.maximum(0, artifact["pipeline"].predict(frame[FEATURES])) * meta["fx"]["eur_to_gbp"]
    frame["preseason_value_gbp"] = frame.preseason_value_million_eur * 1e6 * meta["fx"]["eur_to_gbp"]
    frame["predates_training_cutoff"] = False
    frame["missing_numeric_features"] = frame[NUMERIC].isna().sum(axis=1)
    frame["estimate_available"] = True
    frame["statistics_source"] = "ESPN"
    # The oldest contributing HTTP response controls expiry, not the assembly time.
    fetched_at = min(item["fetched_at"] for item in client.manifest)
    frame["fetched_at"] = fetched_at
    fields = ["player_id", "player_name", "observation_date", "season_start_year", "position", "competition_code", "league", "club",
              "age", "minutes_played", "goals", "assists", "appearances", "goals_per90", "assists_per90", "predicted_market_value_gbp",
              "preseason_value_gbp", "preseason_valuation_date", "preseason_value_missing", "predates_training_cutoff",
              "missing_numeric_features", "estimate_available", "statistics_source", "fetched_at"]
    coverage.update(roster_players=len(athletes), detailed_candidates=len(candidates), omitted_statistics=len(omitted),
                    club_warnings=warnings)
    result = dict(status="experimental", fetched_at=fetched_at, source=f"https://www.espn.com/soccer/league/_/name/{slug}",
                  competition_code=code, coverage=coverage, fx=meta["fx"], historical_metrics=meta["metrics"],
                  historical_elite_metrics=meta["value_bands"]["elite_50m_plus"],
                  model_sha256=hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
                  players=json.loads(frame[fields].to_json(orient="records")))
    result = expand_squad(result, espn_squad(athletes, teams, elements), players, values, code, name, "ESPN", now)
    mapping.to_csv(archive / "player_mapping.csv", index=False)
    atomic_write(archive / "metadata.json", json.dumps(dict(result, players=None, responses=client.manifest, omitted=omitted), indent=2))
    atomic_write(cache_path(code), json.dumps(result, ensure_ascii=False, allow_nan=False))
    print(f"{name}: {len(result['players'])} squad players, {len(frame)} model estimates saved.", flush=True)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", choices=list(LEAGUES))
    args = parser.parse_args()
    for code in ([args.league] if args.league else LEAGUES):
        try:
            refresh_league(code)
        except (requests.RequestException, ValueError, KeyError, OSError) as exc:
            print(f"{code}: refresh failed ({type(exc).__name__}): {exc}", flush=True)
            raise
