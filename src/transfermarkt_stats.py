"""Prepare auditable real player-season tables from the published Transfermarkt data."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .api_client import atomic_write
from .features import build_features, dates, numbers, require_columns
from .transfermarkt import DATASET_PAGE, VALUE_SOURCE

LEAGUES = ["GB1", "ES1", "L1", "IT1", "FR1"]
STAT_COLUMNS = ["minutes_played", "goals", "assists", "yellow_cards", "red_cards"]


def prepare_seasons(players: pd.DataFrame, games: pd.DataFrame, appearances: pd.DataFrame,
                    valuations: pd.DataFrame, seasons: list[int], competitions: list[str] = LEAGUES,
                    min_minutes: int = 450, max_label_days: int = 90) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Coverage means appearances for selected league fixtures, not every career match.

    Two-game round-robin structure is checked. Clubs affected by incomplete match
    appearances exclude their entire player-season, including transferred players.
    """
    if min_minutes < 0 or max_label_days < 1 or not seasons or len(set(seasons)) != len(seasons):
        raise ValueError("Use distinct seasons, nonnegative min_minutes and positive max_label_days.")
    require_columns(players, ["player_id", "name", "date_of_birth", "position"], "Players")
    require_columns(games, ["game_id", "season", "competition_id", "date", "home_club_id", "away_club_id",
                            "home_club_goals", "away_club_goals"], "Games")
    require_columns(appearances, ["game_id", "player_id", "player_club_id", "date"] + STAT_COLUMNS, "Appearances")
    selected = games.loc[games.season.isin(seasons) & games.competition_id.isin(competitions)].copy()
    if selected.empty or selected.game_id.duplicated().any():
        raise ValueError("No selected games or duplicate game IDs.")
    selected["date"] = dates(selected.date.astype(str).str[:10], "game date")
    coverage = []
    for season in sorted(seasons):
        for competition in competitions:
            group = selected.loc[(selected.season == season) & (selected.competition_id == competition)]
            clubs = set(group.home_club_id) | set(group.away_club_id)
            expected = len(clubs) * (len(clubs) - 1)
            pairs = set(zip(group.home_club_id, group.away_club_id))
            if (len(clubs) < 2 or len(group) != expected or len(pairs) != expected
                    or any(home == away for home, away in pairs)
                    or group[["home_club_goals", "away_club_goals"]].isna().any().any()):
                raise ValueError(f"Incomplete double-round-robin fixtures for {competition} {season}.")
            if group.date.min() < pd.Timestamp(f"{season}-07-01") or group.date.max() > pd.Timestamp(f"{season+1}-06-30"):
                raise ValueError("Selected seasons must follow July–June boundaries.")
            coverage.append({"season": season, "competition": competition, "clubs": len(clubs),
                             "games": len(group), "expected_games": expected})
    # Only match identity and in-game counts enter the statistical aggregation.
    stats = appearances.loc[appearances.game_id.isin(selected.game_id),
                            ["game_id", "player_id", "player_club_id", "date"] + STAT_COLUMNS].copy()
    if stats.duplicated(["game_id", "player_id"]).any():
        raise ValueError("Duplicate player/match appearances must be resolved before aggregation.")
    stats = stats.rename(columns={"date": "appearance_date"}).merge(
        selected[["game_id", "season", "competition_id", "date", "home_club_id", "away_club_id"]],
        on="game_id", validate="many_to_one")
    if not dates(stats.appearance_date.astype(str).str[:10], "appearance date").eq(stats.date).all():
        raise ValueError("Appearance dates disagree with fixture dates.")
    if not (stats.player_club_id.eq(stats.home_club_id) | stats.player_club_id.eq(stats.away_club_id)).all():
        raise ValueError("Appearance club is not a participant in the fixture.")
    for col in STAT_COLUMNS:
        stats[col] = numbers(stats[col], col, nullable=False, integer=True)
    team_counts = stats.groupby(["game_id", "player_club_id"]).size()
    affected, bad_games = set(), []
    for game in selected.itertuples():
        if any(team_counts.get((game.game_id, club), 0) < 11 for club in (game.home_club_id, game.away_club_id)):
            affected.update([(game.season, game.home_club_id), (game.season, game.away_club_id)])
            bad_games.append(int(game.game_id))
    bad_rows = [tuple(pair) in affected for pair in stats[["season", "player_club_id"]].to_numpy()]
    excluded_keys = stats.loc[bad_rows, ["player_id", "season"]].drop_duplicates()
    stats = stats.merge(excluded_keys.assign(excluded=True), on=["player_id", "season"], how="left")
    stats = stats.loc[stats.excluded.isna()].copy()
    # Zero-minute substitute appearances still count as appearances, but per90 handles zero minutes.
    aggregate = stats.groupby(["player_id", "season"]).agg(
        appearances=("game_id", "nunique"), **{col: (col, "sum") for col in STAT_COLUMNS}).reset_index()
    # Main competition is the league with most minutes, deterministic on ties.
    leagues = stats.groupby(["player_id", "season", "competition_id"]).minutes_played.sum().reset_index()
    leagues = leagues.sort_values(["minutes_played", "competition_id"], ascending=[False, True]).drop_duplicates(["player_id", "season"])
    aggregate = aggregate.merge(leagues[["player_id", "season", "competition_id"]], on=["player_id", "season"], validate="one_to_one")
    profiles = players[["player_id", "name", "date_of_birth", "position"]].copy()
    if profiles.player_id.duplicated().any():
        raise ValueError("Duplicate player profiles.")
    aggregate = aggregate.merge(profiles, on="player_id", how="left", validate="many_to_one")
    before = len(aggregate)
    aggregate = aggregate.dropna(subset=["name", "date_of_birth", "position"])
    valid_birth = pd.to_datetime(aggregate.date_of_birth.astype(str).str[:10], errors="coerce")
    window_ends = selected.groupby("season").date.max()
    aggregate["window_end"] = aggregate.season.map(window_ends)
    aggregate["observation_date"] = aggregate.window_end + pd.Timedelta(1, unit="D")
    age = (aggregate.observation_date - valid_birth).dt.days / 365.2425
    aggregate = aggregate.loc[age.between(14, 60)].copy()
    aggregate["date_of_birth"] = aggregate.date_of_birth.astype(str).str[:10]
    aggregate["window_start"] = aggregate.season.astype(str) + "-07-01"
    for col in ("window_end", "observation_date"):
        aggregate[col] = aggregate[col].dt.strftime("%Y-%m-%d")
    observations = aggregate.rename(columns={"season": "season_start_year", "competition_id": "competition_code", "name": "player_name"})
    observations["starts"] = np.nan  # Appearance records cannot establish starting-XI membership.
    observations["coverage_complete"] = True
    observations["source"] = VALUE_SOURCE + "; selected domestic leagues only; fixture/team appearance checks passed"
    observations["is_synthetic"] = False
    observations["player_id_namespace"] = "transfermarkt"
    features = build_features(observations)

    require_columns(valuations, ["player_id", "date", "market_value_in_eur"], "Valuations")
    labels = valuations[["player_id", "date", "market_value_in_eur"]].rename(
        columns={"date": "valuation_date", "market_value_in_eur": "market_value_eur"}).copy()
    labels["valuation_date"] = dates(labels.valuation_date.astype(str).str[:10], "valuation date")
    labels["market_value_eur"] = numbers(labels.market_value_eur, "market_value_eur", nullable=False)
    if labels.duplicated(["player_id", "valuation_date"]).any():
        raise ValueError("Duplicate player/date valuations.")
    eligible = features.loc[features.minutes_played >= min_minutes, ["player_id", "observation_date"]].copy()
    eligible["observation_date"] = dates(eligible.observation_date, "observation date")
    joined = pd.merge_asof(eligible.sort_values("observation_date"), labels.sort_values("valuation_date"),
                           by="player_id", left_on="observation_date", right_on="valuation_date",
                           direction="forward", allow_exact_matches=False,
                           tolerance=pd.Timedelta(max_label_days, unit="D"))
    matched = joined.valuation_date.notna()
    labels = joined.loc[matched, ["player_id", "valuation_date", "market_value_eur"]].drop_duplicates()
    if labels.empty:
        raise ValueError("No historical valuation labels within the specified horizon.")
    labels["valuation_date"] = labels.valuation_date.dt.strftime("%Y-%m-%d")
    labels["value_source"] = VALUE_SOURCE
    labels["is_synthetic"] = False
    labels["player_id_namespace"] = "transfermarkt"
    report = {"source": DATASET_PAGE, "player_id_namespace": "transfermarkt", "seasons": seasons,
              "competitions": competitions, "fixture_coverage": coverage, "missing_appearance_game_ids": bad_games,
              "excluded_affected_player_seasons": len(excluded_keys), "invalid_profile_player_seasons": before-len(aggregate),
              "feature_rows": len(features), "eligible_min_minutes": min_minutes, "eligible_rows": len(eligible),
              "label_rows": len(labels), "unmatched_eligible_observations": int((~matched).sum()),
              "max_label_days": max_label_days, "starts_available": False,
              "limitations": ["Selected-league appearances only, not all career competitions.",
                              "Profile positions are from the downloaded snapshot, not historical archives.",
                              "Fixture and team-level coverage checks cannot prove every individual record is correct.",
                              "Players with no recorded appearances are outside this sample."]}
    return features, labels, report


def prepare_from_files(input_dir: Path, output_dir: Path, seasons: list[int],
                       competitions: list[str] = LEAGUES, min_minutes: int = 450, max_label_days: int = 90) -> dict:
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if output_dir.exists():
        raise ValueError("Output directory exists; choose a new one to preserve the prepared dataset.")
    frames = [pd.read_csv(input_dir / f"{name}.csv.gz") for name in ("players", "games", "appearances", "player_valuations")]
    features, labels, report = prepare_seasons(*frames, seasons, competitions, min_minutes, max_label_days)
    metadata = input_dir / "download_metadata.json"
    if metadata.exists():
        report["download_provenance"] = json.loads(metadata.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True)
    atomic_write(output_dir / "features.csv", features.to_csv(index=False))
    atomic_write(output_dir / "valuations.csv", labels.to_csv(index=False))
    atomic_write(output_dir / "preparation_report.json", json.dumps(report, indent=2) + "\n")
    return report
