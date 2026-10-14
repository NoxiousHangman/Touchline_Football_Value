"""CLI workflows for feature preparation, Transfermarkt labels and modelling."""

import json
from pathlib import Path

import pandas as pd

from .api_client import atomic_write
from .config import PROCESSED_DIR, RAW_DIR

PIPELINE_COMMANDS = {"features", "train", "predict", "demo", "download-transfermarkt", "mapping-candidates",
                     "transfermarkt-labels", "prepare-transfermarkt", "players", "status", "validate-season"}


def add_pipeline_commands(commands) -> None:
    commands.add_parser("status", help="show real-data and model readiness without exposing credentials")
    child = commands.add_parser("validate-season", help="check readiness or score a frozen future-season evaluation")
    child.add_argument("--plan", type=Path, required=True)
    child.add_argument("--input-dir", type=Path, default=RAW_DIR / "transfermarkt")
    child.add_argument("--output-dir", type=Path, required=True, help="new directory for audit/results; never overwrites")
    child = commands.add_parser("prepare-transfermarkt", help="build real season features and labels from published statistics")
    child.add_argument("--input-dir", type=Path, default=RAW_DIR / "transfermarkt")
    child.add_argument("--output-dir", type=Path, default=PROCESSED_DIR / "transfermarkt-real")
    child.add_argument("--seasons", type=int, nargs="+", default=[2023, 2024, 2025])
    child.add_argument("--competitions", nargs="+", default=["GB1", "ES1", "L1", "IT1", "FR1"])
    child.add_argument("--min-minutes", type=int, default=450)
    child.add_argument("--max-label-days", type=int, default=90)
    child = commands.add_parser("players", help="find names and IDs in a prepared feature table")
    child.add_argument("--features", type=Path, required=True)
    child.add_argument("--search", default="")
    child = commands.add_parser("features", help="validate snapshots and calculate features")
    source = child.add_mutually_exclusive_group(required=True)
    source.add_argument("--observations", type=Path)
    source.add_argument("--manifest", type=Path, help="reviewed Phase 1 cache manifest")
    child.add_argument("--output", type=Path, default=PROCESSED_DIR / "features.csv")
    child = commands.add_parser("train", help="train/evaluate on earlier/later valuation dates")
    child.add_argument("--features", type=Path, required=True)
    child.add_argument("--valuations", type=Path, required=True)
    child.add_argument("--valuation-history", type=Path, help="Transfermarkt player_valuations.csv.gz; adds strictly preseason values")
    child.add_argument("--run-name")
    child.add_argument("--test-fraction", type=float, default=.25, help="fraction of distinct dates reserved for testing")
    child.add_argument("--test-from", help="explicit first test valuation date (YYYY-MM-DD)")
    child.add_argument("--max-age-days", type=int, default=180)
    child.add_argument("--eur-to-gbp", type=float, help="pounds per euro; overrides the dated default reference rate")
    child = commands.add_parser("predict", help="predict from a trusted local model")
    child.add_argument("--features", type=Path, required=True)
    child.add_argument("--model", type=Path, required=True, help="trusted model.joblib produced by this project")
    child.add_argument("--valuation-history", type=Path, help="required for models trained with preseason market history")
    child.add_argument("--player-id", type=int, help="predict the latest supplied observation for this player")
    child.add_argument("--output", type=Path, default=PROCESSED_DIR / "predictions.csv")
    child.add_argument("--eur-to-gbp", type=float, help="override the model's saved GBP conversion rate")
    child = commands.add_parser("demo", help="run all stages with clearly labelled invented data")
    child.add_argument("--run-name")
    child.add_argument("--eur-to-gbp", type=float, help="override the dated default GBP conversion rate")
    child = commands.add_parser("download-transfermarkt", help="download the published player/valuation snapshot")
    child.add_argument("--output-dir", type=Path, default=RAW_DIR / "transfermarkt")
    child.add_argument("--include-statistics", action="store_true", help="also download games and player appearances")
    child = commands.add_parser("mapping-candidates", help="suggest ID matches for human review")
    child.add_argument("--features", type=Path, required=True)
    child.add_argument("--players", type=Path, default=RAW_DIR / "transfermarkt" / "players.csv.gz")
    child.add_argument("--output", type=Path, default=PROCESSED_DIR / "player_id_candidates.csv")
    child = commands.add_parser("transfermarkt-labels", help="import historical labels using reviewed player IDs")
    child.add_argument("--features", type=Path, required=True)
    child.add_argument("--mapping", type=Path, required=True)
    child.add_argument("--valuations", type=Path, default=RAW_DIR / "transfermarkt" / "player_valuations.csv.gz")
    child.add_argument("--max-delay-days", type=int, default=180)
    child.add_argument("--output", type=Path, default=PROCESSED_DIR / "valuations.csv")


def save_frame(frame: pd.DataFrame, path: Path) -> None:
    atomic_write(path, frame.to_csv(index=False))
    print(f"Saved {len(frame)} rows: {path}")


def show_training(result: dict) -> None:
    report = result["report"]
    if report["is_synthetic"]:
        print("SYNTHETIC DEMO ONLY — scores below do not measure real football accuracy.")
    score = report["metrics"]
    print(f"Train: {report['train_rows']} rows; holdout: {report['test_rows']} rows")
    print(f"Holdout MAE: £{score['mae_gbp']:,.0f}; RMSE: £{score['rmse_gbp']:,.0f}; R2: {score['r2']}")
    print(f"Training-median baseline MAE: £{report['median_baseline_metrics']['mae_gbp']:,.0f}")
    elite = report.get("value_band_metrics", {}).get("elite_50m_plus", {})
    if elite.get("rows"):
        print(f"Players valued at £50m+: MAE £{elite['mae_gbp']:,.0f} across {elite['rows']} examples")
    if report.get("preseason_persistence_metrics"):
        print(f"Preseason-value benchmark MAE: £{report['preseason_persistence_metrics']['mae_gbp']:,.0f}")
    print(f"GBP conversion: {report['fx']['eur_to_gbp']} pounds per euro (saved in model metadata)")
    for key in ("model_path", "report_path", "dataset_path", "plot_path"):
        print(f"{key}: {result[key]}")
    for warning in report["warnings"]:
        print(warning)


def run_pipeline_command(args) -> int:
    if args.command == "validate-season":
        from .future_validation import evaluate_frozen
        report = evaluate_frozen(args.plan, args.input_dir, args.output_dir)
        print(f"Validation status: {report['status']}")
        for reason in report["reasons"]:
            print(reason)
        print(f"Evaluation cannot run before {report['not_before']}.")
        print(f"Audit report: {args.output_dir / 'report.json'}")
        if report.get("results"):
            print(json.dumps(report["results"], indent=2))
    elif args.command == "status":
        from .readiness import project_status
        print(json.dumps(project_status(), indent=2))
    elif args.command == "prepare-transfermarkt":
        from .transfermarkt_stats import prepare_from_files
        report = prepare_from_files(args.input_dir, args.output_dir, args.seasons, args.competitions,
                                    args.min_minutes, args.max_label_days)
        print(f"Prepared {report['feature_rows']} real player-season feature rows and {report['label_rows']} valuation labels.")
        print(f"Coverage report: {args.output_dir / 'preparation_report.json'}")
    elif args.command == "players":
        from .features import validate_features
        frame = validate_features(pd.read_csv(args.features)).sort_values("observation_date").drop_duplicates("player_id", keep="last")
        frame = frame.loc[frame.player_name.str.contains(args.search, case=False, regex=False)]
        print(frame[["player_id", "player_name", "player_id_namespace", "observation_date"]].sort_values("player_name").to_string(index=False))
    elif args.command == "features":
        from .features import build_features
        from .cache_import import observations_from_manifest
        observations = observations_from_manifest(args.manifest) if args.manifest else pd.read_csv(args.observations)
        save_frame(build_features(observations), args.output)
    elif args.command == "train":
        from .train import train_model
        result = train_model(pd.read_csv(args.features), pd.read_csv(args.valuations), run_name=args.run_name,
                              test_fraction=args.test_fraction, test_from=args.test_from, max_age_days=args.max_age_days,
                              eur_to_gbp=args.eur_to_gbp,
                              valuation_history=pd.read_csv(args.valuation_history) if args.valuation_history else None)
        show_training(result)
    elif args.command == "predict":
        from .predict import load_model, predict_players
        artifact = load_model(args.model)
        predictions = predict_players(pd.read_csv(args.features), artifact, args.player_id, args.eur_to_gbp,
                                      valuation_history=pd.read_csv(args.valuation_history) if args.valuation_history else None)
        save_frame(predictions, args.output)
        if artifact["metadata"]["is_synthetic"]:
            print("SYNTHETIC DEMO ONLY — these are not real player estimates.")
        if args.player_id is not None:
            row = predictions.iloc[0]
            print(f"{row.player_name}: estimated market value £{row.predicted_market_value_gbp:,.0f}")
        print("Contributions are additive model terms, not causal effects or confidence intervals.")
    elif args.command == "demo":
        from .demo import demo_data
        from .features import build_features
        from .train import train_model
        from .predict import load_model, predict_players
        observations, valuations = demo_data()
        features = build_features(observations)
        result = train_model(features, valuations, run_name=args.run_name, eur_to_gbp=args.eur_to_gbp)
        directory = result["dataset_path"].parent
        save_frame(observations, directory / "observations.csv")
        save_frame(features, directory / "features.csv")
        save_frame(valuations, directory / "valuations.csv")
        save_frame(predict_players(features, load_model(result["model_path"]), player_id=1), directory / "sample_prediction.csv")
        show_training(result)
    elif args.command == "download-transfermarkt":
        from .transfermarkt import download_snapshot
        metadata = download_snapshot(args.output_dir, args.include_statistics)
        for name, details in metadata["files"].items():
            print(f"Downloaded {name}: {details['bytes']:,} bytes")
        print(f"Source metadata: {args.output_dir / 'download_metadata.json'}")
    elif args.command == "mapping-candidates":
        from .transfermarkt import mapping_candidates
        save_frame(mapping_candidates(pd.read_csv(args.features), pd.read_csv(args.players)), args.output)
        print("Candidates are unverified. Review identities, resolve ambiguity and record evidence before importing labels.")
    elif args.command == "transfermarkt-labels":
        from .transfermarkt import historical_labels
        labels, report = historical_labels(pd.read_csv(args.features), pd.read_csv(args.valuations),
                                           pd.read_csv(args.mapping), args.max_delay_days)
        save_frame(labels, args.output)
        report_path = args.output.with_suffix(".report.json")
        atomic_write(report_path, json.dumps(report, indent=2) + "\n")
        print(f"Matched {report['matched_observations']} observations; {report['unmatched_observations']} unmatched. Report: {report_path}")
    return 0
