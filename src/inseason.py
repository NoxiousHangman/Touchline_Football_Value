"""Separate partial-season model; never modifies the frozen full-season baseline.

Historical experiment: python -m src.inseason --output-dir models/inseason-v1
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from .config import RAW_DIR
from .currency import gbp_conversion
from .features import dates, numbers, require_columns, CATEGORICAL_FEATURES
from .market_history import add_market_history, HISTORY_FEATURES
from .train import make_pipeline, metrics, value_band_metrics
from .transfermarkt_stats import LEAGUES, STAT_COLUMNS

NUMERIC = ["age", "season_days", "minutes_played"] + [
    f"{name}_per90_shrunk" for name in ("goals", "yellow_cards", "red_cards")
] + HISTORY_FEATURES
FEATURES = NUMERIC + CATEGORICAL_FEATURES
MIN_MINUTES = 180
MAX_LABEL_DAYS = 45


def snapshot(players, games, appearances, valuations, as_of: str):
    """All statistics precede as_of; market input strictly precedes July 1.

    Completeness is checked against supplied finished fixtures. Upstream fixture
    omission and historical profile changes cannot be established from this dump.
    """
    cutoff = dates(pd.Series([as_of]), "as_of").iloc[0]
    season = cutoff.year if cutoff.month >= 7 else cutoff.year - 1
    start = pd.Timestamp(f"{season}-07-01")
    if not 60 <= (cutoff - start).days <= 300:
        raise ValueError("In-season snapshots support September through April only.")
    require_columns(games, ["game_id", "date", "season", "competition_id", "home_club_id", "away_club_id",
                            "home_club_goals", "away_club_goals"], "Games")
    require_columns(appearances, ["game_id", "player_id", "player_club_id", "date"] + STAT_COLUMNS, "Appearances")
    selected = games.loc[(games.season == season) & games.competition_id.isin(LEAGUES) &
                         (games.date.astype(str).str[:10] < as_of)].copy()
    if selected.empty or set(selected.competition_id) != set(LEAGUES):
        raise ValueError("Need finished fixtures for all five selected leagues before this snapshot.")
    selected["date"] = dates(selected.date.astype(str).str[:10], "game date")
    if selected.game_id.duplicated().any() or (selected.date < start).any():
        raise ValueError("Duplicate fixtures or game dates outside the season.")
    if selected[["home_club_goals", "away_club_goals"]].isna().any().any():
        raise ValueError("Unfinished fixtures cannot enter a historical snapshot.")
    stale = (cutoff - selected.groupby("competition_id").date.max()).dt.days > 21
    if stale.any():
        raise ValueError("Fixture coverage is stale for the requested date; no current estimate can be made.")
    stats = appearances.loc[appearances.game_id.isin(selected.game_id),
                            ["game_id", "player_id", "player_club_id", "date"] + STAT_COLUMNS].copy()
    if stats.duplicated(["game_id", "player_id"]).any():
        raise ValueError("Duplicate player/match records.")
    stats = stats.rename(columns={"date": "appearance_date"}).merge(
        selected[["game_id", "date", "competition_id", "home_club_id", "away_club_id"]],
        on="game_id", validate="many_to_one")
    if not dates(stats.appearance_date.astype(str).str[:10], "appearance date").eq(stats.date).all():
        raise ValueError("Appearance and fixture dates disagree.")
    if not (stats.player_club_id.eq(stats.home_club_id) | stats.player_club_id.eq(stats.away_club_id)).all():
        raise ValueError("Appearance club does not match fixture.")
    for name in STAT_COLUMNS:
        stats[name] = numbers(stats[name], name, nullable=False, integer=True)
    counts = stats.groupby(["game_id", "player_club_id"]).size()
    affected = set()
    bad_games = []
    for game in selected.itertuples():
        if any(counts.get((game.game_id, club), 0) < 11 for club in (game.home_club_id, game.away_club_id)):
            affected.update([game.home_club_id, game.away_club_id])
            bad_games.append(int(game.game_id))
    excluded = stats.loc[stats.player_club_id.isin(affected), "player_id"].unique()
    stats = stats.loc[~stats.player_id.isin(excluded)]
    aggregate = stats.groupby("player_id").agg(appearances=("game_id", "nunique"),
                                                **{name: (name, "sum") for name in STAT_COLUMNS}).reset_index()
    leagues = stats.groupby(["player_id", "competition_id"]).minutes_played.sum().reset_index()
    leagues = leagues.sort_values(["minutes_played", "competition_id"], ascending=[False, True]).drop_duplicates("player_id")
    aggregate = aggregate.merge(leagues[["player_id", "competition_id"]], on="player_id", validate="one_to_one")
    require_columns(players, ["player_id", "name", "date_of_birth", "position"], "Players")
    if players.player_id.duplicated().any():
        raise ValueError("Duplicate player profiles.")
    aggregate = aggregate.merge(players[["player_id", "name", "date_of_birth", "position"]], on="player_id", how="left", validate="one_to_one")
    birth = pd.to_datetime(aggregate.date_of_birth.astype(str).str[:10], errors="coerce")
    aggregate["age"] = (cutoff - birth).dt.days / 365.2425
    eligible = aggregate.minutes_played.ge(MIN_MINUTES) & aggregate.age.between(14, 60) & aggregate[["name", "position"]].notna().all(axis=1)
    aggregate = aggregate.loc[eligible].rename(columns={"name": "player_name", "competition_id": "competition_code"}).copy()
    if aggregate.empty:
        raise ValueError("No players have sufficient recorded minutes and valid profiles.")
    aggregate["window_start"] = start.strftime("%Y-%m-%d")
    aggregate["observation_date"] = as_of
    aggregate["season_start_year"] = season
    aggregate["season_days"] = (cutoff - start).days
    aggregate["player_id_namespace"] = "transfermarkt"
    aggregate["is_synthetic"] = False
    for name in ("goals", "assists", "yellow_cards", "red_cards"):
        # Fixed 450-minute denominator regularisation limits short-sample spikes.
        aggregate[f"{name}_per90_shrunk"] = aggregate[name] * 90 / (aggregate.minutes_played + 450)
    aggregate = add_market_history(aggregate, valuations)
    return aggregate, {"as_of": as_of, "rows": len(aggregate), "games": len(selected),
                       "excluded_players_for_missing_appearances": len(excluded), "missing_appearance_games": bad_games}


def attach_labels(frame, valuations):
    labels = valuations[["player_id", "date", "market_value_in_eur"]].copy()
    labels["date"] = dates(labels.date.astype(str).str[:10], "valuation date")
    labels["market_value_in_eur"] = numbers(labels.market_value_in_eur, "market value", nullable=False)
    if labels.duplicated(["player_id", "date"]).any():
        raise ValueError("Duplicate player/date labels.")
    left = frame.copy()
    left["observation_date"] = dates(left.observation_date, "observation date")
    joined = pd.merge_asof(left.sort_values("observation_date"), labels.sort_values("date"),
                           by="player_id", left_on="observation_date", right_on="date",
                           direction="forward", allow_exact_matches=False, tolerance=pd.Timedelta(MAX_LABEL_DAYS, unit="D"))
    matched = joined.date.notna()
    joined = joined.loc[matched].rename(columns={"date": "valuation_date", "market_value_in_eur": "market_value_eur"})
    for name in ("observation_date", "valuation_date"):
        joined[name] = joined[name].dt.strftime("%Y-%m-%d")
    return joined, int((~matched).sum())


def run_experiment(input_dir: Path, output_dir: Path):
    if output_dir.exists():
        raise ValueError("Output directory exists; choose a new run name.")
    tables = {name: pd.read_csv(input_dir / f"{name}.csv.gz") for name in ("players", "games", "appearances", "player_valuations")}
    snapshots, audit = [], []
    # Fixed dates, feature set and split established before evaluating candidates.
    for season in (2023, 2024, 2025):
        for day in (f"{season}-10-01", f"{season}-12-01", f"{season+1}-03-01"):
            frame, report = snapshot(tables["players"], tables["games"], tables["appearances"], tables["player_valuations"], day)
            labelled, unmatched = attach_labels(frame, tables["player_valuations"])
            report.update(label_rows=len(labelled), unmatched_rows=unmatched)
            snapshots.append(labelled)
            audit.append(report)
    data = pd.concat(snapshots, ignore_index=True)
    if data.duplicated(["player_id", "valuation_date"]).any():
        raise ValueError("Repeated target valuation across snapshots; resolve before training.")
    train = data.loc[data.valuation_date < "2025-07-01"]
    test = data.loc[data.valuation_date >= "2025-07-01"]
    if len(train) < 100 or len(test) < 100:
        raise ValueError("Insufficient training or test rows.")
    pipeline = make_pipeline(NUMERIC)
    pipeline.fit(train[FEATURES], train.market_value_eur)
    fx = gbp_conversion()
    rate = fx["eur_to_gbp"]
    predicted = np.maximum(0, pipeline.predict(test[FEATURES])) * rate
    prior = test.preseason_value_million_eur.mul(1_000_000).fillna(train.market_value_eur.median()) * rate
    actual = test.market_value_eur * rate
    report = {"kind": "inseason-v1", "created_at": datetime.now(timezone.utc).isoformat(),
              "live_ready": False, "status": "historical_backtest_only", "train_rows": len(train), "test_rows": len(test),
              "protocol_note": "Exploratory backtest. Cutoffs moved to precede winter/spring valuation updates after sparse-label inspection; no independent test claim.",
              "train_date_max": train.valuation_date.max(), "test_date_min": test.valuation_date.min(),
              "test_date_max": test.valuation_date.max(), "numeric_features": NUMERIC,
              "fx": fx, "min_minutes": MIN_MINUTES, "max_label_days": MAX_LABEL_DAYS,
              "metrics": metrics(actual, predicted), "value_bands": value_band_metrics(actual, predicted),
              "preseason_benchmark": metrics(actual, prior), "preseason_value_bands": value_band_metrics(actual, prior),
              "per_snapshot": {}, "audit": audit, "versions": {"sklearn": sklearn.__version__, "pandas": pd.__version__},
              "source_sha256": {name: hashlib.sha256((input_dir / f"{name}.csv.gz").read_bytes()).hexdigest() for name in tables},
              "limitations": ["Historical backtest, not independent future-season validation.",
                              "Snapshot player profiles may reflect later position corrections.",
                              "Coverage checks cannot detect omitted upstream fixtures.",
                              "Players with fewer than 180 minutes or no future label are outside the evaluation.",
                              "Refresh requires current five-league statistics and correctly matched player IDs.",
                              "Current 2026/27 feeds and access rights have not yet been verified."]}
    heldout = test.copy()
    heldout["predicted_market_value_gbp"] = predicted
    heldout["market_value_gbp"] = actual
    heldout["preseason_benchmark_gbp"] = prior
    for day, group in heldout.groupby("observation_date"):
        report["per_snapshot"][day] = {"rows": len(group), "model": metrics(group.market_value_gbp, group.predicted_market_value_gbp),
                                      "preseason_benchmark": metrics(group.market_value_gbp, group.preseason_benchmark_gbp)}
    output_dir.mkdir(parents=True)
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    joblib.dump({"kind": "inseason-v1", "pipeline": pipeline, "metadata": report}, output_dir / "model.joblib", compress=3)
    data.to_csv(output_dir / "training_dataset.csv", index=False)
    heldout.to_csv(output_dir / "holdout_predictions.csv", index=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=RAW_DIR / "transfermarkt")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run_experiment(args.input_dir, args.output_dir)
    print(json.dumps({key: report[key] for key in ("status", "train_rows", "test_rows", "metrics", "preseason_benchmark", "value_bands")}, indent=2))


if __name__ == "__main__":
    main()
