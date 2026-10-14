"""Build observations from individually reviewed Phase 1 player-match caches."""

import json
from pathlib import Path
import re

import pandas as pd

from .features import OBSERVATION_COLUMNS, boolean_column, require_columns, validate_observations

MANIFEST_COLUMNS = ["cache_file", "season_start_year", "observation_date", "coverage_complete"]
API_COUNTS = {"matchesOnPitch": "appearances", "startingXI": "starts", "minutesPlayed": "minutes_played",
              "goals": "goals", "assists": "assists", "yellowCards": "yellow_cards", "redCards": "red_cards"}


def observations_from_manifest(path: Path) -> pd.DataFrame:
    """Coverage is a human-reviewed assertion; limited payloads alone cannot prove it."""
    path = Path(path)
    manifest = pd.read_csv(path)
    require_columns(manifest, MANIFEST_COLUMNS, "Cache manifest")
    if not boolean_column(manifest.coverage_complete, "coverage_complete").all():
        raise ValueError("Review each cache's full-window coverage before setting coverage_complete=true.")
    rows = []
    for entry in manifest.to_dict("records"):
        cache_path = Path(str(entry["cache_file"]))
        if not cache_path.is_absolute():
            cache_path = path.parent / cache_path
        envelope = json.loads(cache_path.read_text(encoding="utf-8"))
        request = envelope.get("request", {})
        match = re.fullmatch(r"https://api\.football-data\.org/v4/persons/(\d+)/matches", request.get("url", ""))
        if not match or envelope.get("source") != "football-data.org":
            raise ValueError(f"{cache_path.name}: expected a Phase 1 football-data.org player-match cache.")
        params = request.get("params", {})
        if not all(params.get(key) for key in ("dateFrom", "dateTo", "competitions")):
            raise ValueError(f"{cache_path.name}: explicit dateFrom, dateTo and one competition are required.")
        if "," in str(params["competitions"]) or int(params.get("offset", 0)) != 0:
            raise ValueError("Use one competition and the initial match window; offset fragments cannot be season totals.")
        payload = envelope.get("data", {})
        games, aggregate, person = payload.get("matches"), payload.get("aggregations"), payload.get("person", {})
        if not isinstance(games, list) or not isinstance(aggregate, dict):
            raise ValueError(f"{cache_path.name}: missing matches or aggregations.")
        if len(games) >= int(params.get("limit", 15)):
            raise ValueError("Match count reaches the request limit: coverage may be truncated. Review a larger window limit.")
        if str(person.get("id")) != match.group(1):
            raise ValueError("Player ID in payload disagrees with the request.")
        ids = []
        for game in games:
            if not isinstance(game, dict) or game.get("id") is None:
                raise ValueError("Each returned match must have an ID.")
            ids.append(game["id"])
            played = str(game.get("utcDate", ""))[:10]
            if game.get("status") != "FINISHED" or not params["dateFrom"] <= played <= params["dateTo"]:
                raise ValueError("All returned matches must be finished and inside the requested dates.")
            competition = game.get("competition", {})
            if str(params["competitions"]) not in (str(competition.get("code")), str(competition.get("id"))):
                raise ValueError("A returned match is outside the requested competition.")
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate matches in cache; do not sum duplicated aggregates.")
        if aggregate.get("matchesOnPitch") is not None and aggregate["matchesOnPitch"] > len(games):
            raise ValueError("Aggregate appearances exceed returned matches.")
        rows.append({"player_id": int(person["id"]), "player_name": person.get("name"),
                     "date_of_birth": person.get("dateOfBirth"), "position": person.get("position"),
                     "competition_code": str(params["competitions"]), "season_start_year": entry["season_start_year"],
                     "window_start": params["dateFrom"], "window_end": params["dateTo"],
                     "observation_date": entry["observation_date"], "coverage_complete": True,
                     "source": str(cache_path.resolve()), "is_synthetic": False,
                     **{column: aggregate.get(key) for key, column in API_COUNTS.items()}})
    return validate_observations(pd.DataFrame(rows))[OBSERVATION_COLUMNS]
