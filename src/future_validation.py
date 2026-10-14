"""Freeze a season evaluation in advance; score without fitting or tuning."""

from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd

from .api_client import atomic_write
from .dataset import build_training_dataset
from .predict import load_model, predict_players
from .train import metrics, value_band_metrics
from .transfermarkt_stats import LEAGUES, prepare_seasons

ROOT = Path(__file__).resolve().parents[1]
FROZEN_CODE = ["features.py", "market_history.py", "predict.py", "dataset.py",
               "transfermarkt_stats.py", "train.py", "future_validation.py",
               "currency.py", "api_client.py", "config.py", "transfermarkt.py"]


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plan_digest(plan: dict) -> str:
    return hashlib.sha256(json.dumps({k: v for k, v in plan.items() if k != "plan_sha256"},
                                     sort_keys=True).encode()).hexdigest()


def freeze_validation(model_path: Path, training_dataset: Path, output_dir: Path,
                      season: int = 2026) -> dict:
    """Create a new, auditable plan and separate copy of the trusted fitted model."""
    model_path, training_dataset, output_dir = map(Path, (model_path, training_dataset, output_dir))
    if output_dir.exists():
        raise ValueError("Validation plan directory exists; preserve the original frozen plan.")
    artifact = load_model(model_path)
    meta = artifact["metadata"]
    start = date(season, 7, 1)
    if meta["is_synthetic"] or meta.get("player_id_namespace") != "transfermarkt" or not meta.get("market_history"):
        raise ValueError("Freeze a real Transfermarkt model with preseason history.")
    if start.isoformat() <= max(meta["train_date_max"], meta["test_date_max"]):
        raise ValueError("Evaluation season must follow all training and previously evaluated labels.")
    saved = pd.read_csv(training_dataset)
    train = saved.loc[saved.split == "train"]
    if len(train) != meta["train_rows"] or train.valuation_date.max() != meta["train_date_max"]:
        raise ValueError("Training dataset does not match the frozen model's training rows/dates.")
    plan = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "season_start_year": season, "competitions": LEAGUES,
        "history_seasons": [season - 2, season - 1, season],
        "min_minutes": 450, "max_label_days": 90,
        # Conservative: allow the entire July-June season and label horizon to elapse.
        "not_before": (date(season + 1, 7, 1) + timedelta(days=90)).isoformat(),
        "model_file": "model.joblib", "model_sha256": digest(model_path),
        "model_metadata": meta, "training_dataset_sha256": digest(training_dataset),
        "training_median_eur": float(train.market_value_eur.median()),
        "code_sha256": {name: digest(ROOT / "src" / name) for name in FROZEN_CODE},
        "primary_metrics": ["overall MAE", "MAE for actual value >= GBP 50m"],
        "secondary_metrics": ["RMSE", "R2", "signed bias", "actual value >= GBP 100m"],
        "benchmarks": ["preseason value, training median if missing", "training median"],
        "evaluation_rule": "Report all results; no refitting, calibration, candidate selection or automatic promotion.",
        "independence_note": "2025/26 was already inspected. Only the specified later season is scored.",
    }
    plan["plan_sha256"] = plan_digest(plan)
    output_dir.mkdir(parents=True)
    shutil.copyfile(model_path, output_dir / "model.joblib")
    (output_dir / "source").mkdir()
    for name in FROZEN_CODE:
        shutil.copyfile(ROOT / "src" / name, output_dir / "source" / name)
    atomic_write(output_dir / "plan.json", json.dumps(plan, indent=2, allow_nan=False) + "\n")
    return plan


def evaluate_frozen(plan_path: Path, input_dir: Path, output_dir: Path, *, today: date | None = None) -> dict:
    """Audit readiness or evaluate once; never call fit, overwrite a run, or alter the model."""
    plan_path, input_dir, output_dir = map(Path, (plan_path, input_dir, output_dir))
    if output_dir.exists():
        raise ValueError("Evaluation output directory exists. Use a new directory to preserve results.")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != 1 or plan.get("plan_sha256") != plan_digest(plan):
        raise ValueError("Validation plan changed or has an unsupported schema.")
    model_path = plan_path.parent / plan["model_file"]
    if digest(model_path) != plan["model_sha256"]:
        raise ValueError("Frozen model changed; independent evaluation stopped.")
    for name, expected in plan["code_sha256"].items():
        if digest(ROOT / "src" / name) != expected:
            raise ValueError(f"Evaluation code changed ({name}); restore the frozen version before evaluation.")
    artifact = load_model(model_path)
    if artifact["metadata"] != plan["model_metadata"]:
        raise ValueError("Frozen model metadata differs from the validation plan.")
    today = today or date.today()
    tables = {}
    report = {"status": "waiting_for_data", "checked_on": today.isoformat(),
              "season_start_year": plan["season_start_year"], "not_before": plan["not_before"],
              "plan_sha256": plan["plan_sha256"], "model_sha256": plan["model_sha256"],
              "reasons": [], "source_files": {}, "latest_record_dates": {}}
    for name in ("players", "games", "appearances", "player_valuations"):
        path = input_dir / f"{name}.csv.gz"
        if not path.exists():
            report["reasons"].append(f"Missing {name}.csv.gz")
            continue
        tables[name] = pd.read_csv(path)
        report["source_files"][name] = {"sha256": digest(path), "rows": len(tables[name])}
        if "date" in tables[name]:
            report["latest_record_dates"][name] = str(tables[name].date.max())
    if today.isoformat() < plan["not_before"]:
        report["reasons"].append("The complete season and 90-day valuation window have not yet elapsed.")
    if "games" in tables:
        games = tables["games"]
        selected = games.loc[(games.season == plan["season_start_year"]) & games.competition_id.isin(plan["competitions"])]
        report["target_season_fixture_counts"] = {code: int(selected.competition_id.eq(code).sum()) for code in plan["competitions"]}
        if selected.empty:
            report["reasons"].append("No fixtures for the target season in the selected leagues.")
    if not report["reasons"]:
        # Reject future-dated data rather than silently assessing unavailable outcomes.
        for name in ("games", "appearances", "player_valuations"):
            if (tables[name].date.astype(str).str[:10] > today.isoformat()).any():
                raise ValueError(f"{name} contains records after the evaluation date.")
        try:
            features, labels, coverage = prepare_seasons(
                tables["players"], tables["games"], tables["appearances"], tables["player_valuations"],
                plan["history_seasons"], plan["competitions"], plan["min_minutes"], plan["max_label_days"])
        except ValueError as exc:
            report["reasons"].append(f"Preparation checks not satisfied: {exc}")
        else:
            joined = build_training_dataset(features, labels, plan["max_label_days"])
            target = joined.loc[joined.season_start_year == plan["season_start_year"]].copy()
            report["coverage"] = coverage
            report["target_eligible_rows"] = int(((features.season_start_year == plan["season_start_year"]) &
                                                  (features.minutes_played >= plan["min_minutes"])).sum())
            report["target_label_rows"] = len(target)
            report["target_unmatched_rows"] = report["target_eligible_rows"] - len(target)
            target_features = features.loc[features.season_start_year == plan["season_start_year"]]
            label_window_end = (date.fromisoformat(target_features.observation_date.max()) +
                                timedelta(days=plan["max_label_days"])) if not target_features.empty else None
            history_through = tables["player_valuations"].date.astype(str).str[:10].max()
            if target.empty:
                report["reasons"].append("No eligible labelled players for the target season.")
            elif label_window_end and history_through < label_window_end.isoformat():
                report["reasons"].append("Valuation snapshot does not yet span the complete target label window.")
            else:
                predictions = predict_players(features, artifact, valuation_history=tables["player_valuations"])
                scored = target[["player_id", "player_name", "observation_date", "valuation_date", "market_value_eur"]].merge(
                    predictions.drop(columns="player_name"), on=["player_id", "observation_date"], validate="one_to_one")
                if len(scored) != len(target):
                    raise ValueError("Prediction join lost evaluation rows.")
                rate = plan["model_metadata"]["fx"]["eur_to_gbp"]
                scored["market_value_gbp"] = scored.market_value_eur * rate
                median = plan["training_median_eur"] * rate
                scored["preseason_persistence_prediction_gbp"] = scored.preseason_value_gbp.fillna(median)
                scored["median_prediction_gbp"] = median
                methods = {"frozen_model": scored.predicted_market_value_gbp,
                           "preseason_persistence": scored.preseason_persistence_prediction_gbp,
                           "training_median": scored.median_prediction_gbp}
                report["results"] = {name: {"overall": metrics(scored.market_value_gbp, pred),
                                             "bands": value_band_metrics(scored.market_value_gbp, pred)}
                                     for name, pred in methods.items()}
                report["missing_preseason_history_rows"] = int(scored.preseason_value_missing.sum())
                report["status"] = "evaluated"
                report["limitations"] = ["Retrospective source corrections cannot be ruled out.",
                                           "Players without qualifying labels are excluded and counted.",
                                           "No automatic claim of success; inspect elite sample size and benchmark errors."]
                output_dir.mkdir(parents=True)
                atomic_write(output_dir / "predictions.csv", scored.to_csv(index=False))
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write(output_dir / "report.json", json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
