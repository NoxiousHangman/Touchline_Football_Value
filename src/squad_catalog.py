"""Searchable source squads with explicit model/reference/profile-only tiers."""
from collections import Counter
import json

import numpy as np
import pandas as pd

from .market_history import add_market_history
from .player_identity import match_profiles


def fpl_squad(bootstrap):
    teams = {t["id"]: t["name"] for t in bootstrap.get("teams", [])}
    positions = {1: "Goalkeeper", 2: "Defender", 3: "Midfield", 4: "Attack"}
    return [dict(id=str(a["id"]), name=f"{a['first_name']} {a['second_name']}",
                 aliases=[a.get("web_name"), a.get("known_name")], dob=a.get("birth_date"),
                 position=positions.get(a.get("element_type"), "Unknown"), club=teams[a["team"]],
                 minutes_played=a.get("minutes"), goals=a.get("goals_scored"), appearances=None, assists=None)
            for a in bootstrap["elements"] if a.get("team") in teams and not a.get("removed", False)]


def espn_squad(athletes, teams, detailed):
    clubs = {t["id"]: t["displayName"] for t in teams}
    details = {str(a["id"]): a for a in detailed}
    rows = []
    for a in athletes:
        stats = {s["name"]: s["value"] for c in a.get("statistics", {}).get("splits", {}).get("categories", []) for s in c.get("stats", [])}
        detail = details.get(str(a["id"]), {})
        rows.append(dict(id=str(a["id"]), name=a["fullName"], aliases=[a.get("displayName")],
                         dob=a.get("dateOfBirth"), position=a.get("position", {}).get("displayName", "Unknown"),
                         club=clubs.get(a["club_id"], "Unknown"), minutes_played=detail.get("minutes"),
                         goals=stats.get("totalGoals"), appearances=stats.get("appearances"), assists=stats.get("goalAssists")))
    return rows


def expand_squad(snapshot, roster, players, values, code, league, source, now):
    """Do not extrapolate the trained model below its 180-minute population."""
    roster = [r for r in roster if r.get("name")]
    source_counts = Counter(r["id"] for r in roster)
    roster = [r for r in roster if source_counts[r["id"]] == 1]
    matches = match_profiles(roster, players)
    profiles = players.set_index("player_id").to_dict("index")
    season = snapshot["coverage"]["season"]
    matched_ids = list(set(matches.values()))
    history = pd.DataFrame({"player_id": matched_ids, "player_id_namespace": "transfermarkt", "is_synthetic": False,
                            "window_start": f"{season}-07-01"})
    priors = add_market_history(history, values).set_index("player_id").to_dict("index") if matched_ids else {}
    model_rows = {r["player_id"]: r for r in snapshot["players"] if r.get("estimate_available")}
    result = []
    for r in roster:
        tm_id = matches.get(r["id"])
        profile = profiles.get(tm_id, {})
        prior = priors.get(tm_id, {})
        value = prior.get("preseason_value_million_eur")
        prior_gbp = float(value) * 1e6 * snapshot["fx"]["eur_to_gbp"] if value is not None and pd.notna(value) else None
        model = model_rows.get(tm_id)
        if model:
            record = dict(model, valuation_basis="model", reference_market_value_gbp=None,
                          valuation_value_gbp=model["predicted_market_value_gbp"], coverage_note="Current-season model estimate.")
        else:
            dob = pd.to_datetime(str(r.get("dob", ""))[:10], errors="coerce")
            age = (pd.Timestamp(now.date()) - dob).days / 365.2425 if pd.notna(dob) else None
            if age is not None and not 14 <= age <= 60:
                age = None
            # Negative IDs are local source identities, never Transfermarkt IDs.
            local_id = -(100_000_000 if source == "FPL" else 200_000_000) - int(r["id"])
            note = ("Fewer than 180 verified league minutes; showing the dated preseason value."
                    if r.get("minutes_played") is not None and r["minutes_played"] < 180 else
                    "A current model estimate is unavailable; showing the dated preseason value.")
            if prior_gbp is None:
                note = "Profile only: no safely matched, qualifying preseason valuation. No price has been invented."
            record = dict(player_id=tm_id if tm_id is not None else local_id, player_name=profile.get("name", r["name"]),
                          observation_date=now.date().isoformat(), season_start_year=season, position=profile.get("position", r["position"]),
                          competition_code=code, league=league, club=r["club"], age=age, minutes_played=r.get("minutes_played"),
                          goals=r.get("goals"), assists=r.get("assists"), appearances=r.get("appearances"),
                          goals_per90=None, assists_per90=None, predicted_market_value_gbp=None,
                          preseason_value_gbp=prior_gbp, preseason_valuation_date=prior.get("preseason_valuation_date"),
                          preseason_value_missing=int(prior_gbp is None), predates_training_cutoff=False,
                          missing_numeric_features=None, estimate_available=False,
                          valuation_basis="preseason_reference" if prior_gbp is not None else "profile_only",
                          reference_market_value_gbp=prior_gbp, valuation_value_gbp=prior_gbp, coverage_note=note)
        record.update(source_player_id=r["id"], player_id_namespace="transfermarkt" if tm_id is not None else source.lower(),
                      statistics_source=source, fetched_at=snapshot["fetched_at"], club=r["club"])
        result.append(record)
    # Pandas round-trip converts missing optional profile fields to valid JSON nulls.
    snapshot["players"] = json.loads(pd.DataFrame(result).replace([np.inf, -np.inf], np.nan).to_json(orient="records"))
    counts = Counter(r["valuation_basis"] for r in result)
    snapshot["coverage"].update(squad_players=len(result), model_estimates=counts["model"],
                                preseason_references=counts["preseason_reference"], profile_only=counts["profile_only"])
    return snapshot
