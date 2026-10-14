"""Time-held-out Linear Regression training and reproducible evaluation artifacts."""

from datetime import datetime, timezone
import hashlib
import json
import math
import platform
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .api_client import atomic_write
from .config import MODELS_DIR, PROCESSED_DIR, VISUALISATIONS_DIR
from .dataset import build_training_dataset
from .currency import gbp_conversion
from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, dates
from .market_history import add_market_history, HISTORY_FEATURES, HISTORY_MAX_AGE_DAYS


def make_pipeline(numeric_features: list[str] | None = None) -> Pipeline:
    numeric_features = NUMERIC_FEATURES if numeric_features is None else numeric_features
    preprocessing = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), numeric_features),
        ("category", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False),
         CATEGORICAL_FEATURES),
    ], remainder="drop")
    return Pipeline([("preprocess", preprocessing), ("regression", LinearRegression())])


def chronological_split(data: pd.DataFrame, test_fraction: float = 0.25,
                        test_from: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not math.isfinite(test_fraction) or not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between zero and one.")
    ordered = data.sort_values(["valuation_date", "player_id"])
    unique_dates = sorted(ordered.valuation_date.unique())
    if len(unique_dates) < 2:
        raise ValueError("At least two different valuation dates are required for chronological evaluation.")
    if test_from is None:
        n_test_dates = min(len(unique_dates) - 1, max(1, math.ceil(len(unique_dates) * test_fraction)))
        cutoff = unique_dates[-n_test_dates]
    else:
        dates(pd.Series([test_from]), "test_from")
        cutoff = test_from
    train = ordered.loc[ordered.valuation_date < cutoff].copy()
    test = ordered.loc[ordered.valuation_date >= cutoff].copy()
    if len(train) < 20 or len(test) < 5:
        raise ValueError("Need at least 20 earlier training rows and 5 later test rows. Add data or change the cutoff.")
    return train, test


def metrics(actual, predicted) -> dict:
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    return {"mae_gbp": float(mean_absolute_error(actual, predicted)),
            "rmse_gbp": float(np.sqrt(mean_squared_error(actual, predicted))),
            "r2": float(r2_score(actual, predicted)) if np.ptp(actual) > 0 else None}


def value_band_metrics(actual, predicted) -> dict:
    """Positive bias means overprediction; bands are defined by actual GBP value."""
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    result = {}
    for name, low, high in (("below_50m", 0, 50_000_000),
                            ("elite_50m_plus", 50_000_000, np.inf),
                            ("elite_100m_plus", 100_000_000, np.inf)):
        mask = (actual >= low) & (actual < high)
        result[name] = {"rows": int(mask.sum())}
        if mask.any():
            result[name].update(metrics(actual[mask], predicted[mask]))
            result[name]["mean_prediction_minus_actual_gbp"] = float((predicted[mask] - actual[mask]).mean())
    return result


def train_model(features: pd.DataFrame, valuations: pd.DataFrame, *, run_name: str | None = None,
                test_fraction: float = 0.25, test_from: str | None = None, max_age_days: int = 180,
                models_dir: Path = MODELS_DIR, processed_dir: Path = PROCESSED_DIR,
                visualisations_dir: Path = VISUALISATIONS_DIR, eur_to_gbp: float | None = None,
                valuation_history: pd.DataFrame | None = None) -> dict:
    fx = gbp_conversion(eur_to_gbp)
    rate = fx["eur_to_gbp"]
    data = build_training_dataset(features, valuations, max_age_days)
    numeric_features = NUMERIC_FEATURES.copy()
    history_metadata = None
    if valuation_history is not None:
        data = add_market_history(data, valuation_history)
        numeric_features += HISTORY_FEATURES
        history_metadata = {
            "method": "latest valuation strictly before season window_start",
            "max_age_days": HISTORY_MAX_AGE_DAYS,
            "source": "Transfermarkt via dcaribou/transfermarkt-datasets",
            "history_sha256": hashlib.sha256(valuation_history[
                ["player_id", "date", "market_value_in_eur"]].to_csv(index=False).encode()).hexdigest(),
        }
    model_features = numeric_features + CATEGORICAL_FEATURES
    train, test = chronological_split(data, test_fraction, test_from)
    if history_metadata and train.preseason_value_missing.all():
        raise ValueError("No training rows have usable preseason market history. Check player IDs and dates.")
    pipeline = make_pipeline(numeric_features)
    pipeline.fit(train[model_features], train.market_value_eur)
    raw = pipeline.predict(test[model_features])
    predicted = np.maximum(0, raw)
    baseline = np.repeat(train.market_value_eur.median(), len(test))
    synthetic = bool(data.is_synthetic.iloc[0])
    preprocess = pipeline.named_steps["preprocess"]
    regressor = pipeline.named_steps["regression"]
    feature_names = preprocess.get_feature_names_out().tolist()
    warnings = []
    if synthetic:
        warnings.append("SYNTHETIC DEMO ONLY: these scores do not measure real football valuation accuracy.")
    if len(train) <= len(feature_names) + 1:
        warnings.append("Training sample count is small relative to the fitted feature count.")
    if regressor.rank_ < len(feature_names):
        warnings.append("The design matrix is rank deficient; individual coefficients are not uniquely interpretable.")
    if (raw < 0).any():
        warnings.append("Some raw predictions were negative; displayed estimates use a zero floor.")
    report = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "model": "LinearRegression", "target": "market_value_eur", "model_currency": "EUR",
        "currency": "GBP", "fx": fx, "prediction_upper_cap": None,
        "player_id_namespace": data.player_id_namespace.iloc[0],
        "is_synthetic": synthetic, "train_rows": len(train), "test_rows": len(test),
        "train_date_min": train.valuation_date.min(), "train_date_max": train.valuation_date.max(),
        "test_date_min": test.valuation_date.min(), "test_date_max": test.valuation_date.max(),
        "test_players_seen_in_training": int(test.loc[test.player_id.isin(train.player_id), "player_id"].nunique()),
        "split_method": "strict chronological cutoff by valuation date; no holdout refit",
        "max_feature_age_days": max_age_days, "metrics": metrics(test.market_value_eur * rate, predicted * rate),
        "raw_linear_metrics": metrics(test.market_value_eur * rate, raw * rate),
        "median_baseline_metrics": metrics(test.market_value_eur * rate, baseline * rate),
        "negative_raw_predictions": int((raw < 0).sum()), "numeric_features": numeric_features,
        "market_history": history_metadata,
        "value_band_metrics": value_band_metrics(test.market_value_eur * rate, predicted * rate),
        "categorical_features": CATEGORICAL_FEATURES, "warnings": warnings,
        "entirely_missing_training_features": train[numeric_features].columns[train[numeric_features].isna().all()].tolist(),
        "value_sources": sorted(data.value_source.unique().tolist()),
        "dataset_sha256": hashlib.sha256(data.to_csv(index=False).encode()).hexdigest(),
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
    }
    if report["metrics"]["mae_gbp"] >= report["median_baseline_metrics"]["mae_gbp"]:
        warnings.append("The model did not beat the training-median baseline on holdout MAE.")
    if history_metadata:
        # Persistence is the essential benchmark when using an earlier market value.
        persistence = test.preseason_value_million_eur.mul(1_000_000).fillna(train.market_value_eur.median()) * rate
        report["preseason_persistence_metrics"] = metrics(test.market_value_eur * rate, persistence)
        report["preseason_persistence_value_bands"] = value_band_metrics(test.market_value_eur * rate, persistence)
        report["missing_preseason_history"] = {"train_rows": int(train.preseason_value_missing.sum()),
                                               "test_rows": int(test.preseason_value_missing.sum())}
        warnings.append("This model estimates changes from a preseason market value; it is not a statistics-only valuation.")
        if report["metrics"]["mae_gbp"] >= report["preseason_persistence_metrics"]["mae_gbp"]:
            warnings.append("The model did not beat carrying forward preseason values on overall holdout MAE.")
    name = run_name or ("demo-" if synthetic else "run-") + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name):
        raise ValueError("run_name must contain only letters, numbers, underscores or hyphens (max 80 characters).")
    model_dir, data_dir, plot_dir = (Path(root) / name for root in (models_dir, processed_dir, visualisations_dir))
    if any(path.exists() for path in (model_dir, data_dir, plot_dir)):
        raise ValueError("Run name already exists. Choose a new name to preserve earlier results.")
    for path in (model_dir, data_dir, plot_dir):
        path.mkdir(parents=True)
    heldout = test[["player_id", "player_name", "observation_date", "valuation_date", "age",
                    "market_value_eur", "is_synthetic"]].copy()
    heldout["market_value_gbp"] = heldout.market_value_eur * rate
    heldout["raw_prediction_gbp"] = raw * rate
    heldout["predicted_market_value_gbp"] = predicted * rate
    heldout["baseline_prediction_gbp"] = baseline * rate
    heldout["residual_gbp"] = (heldout.market_value_eur - predicted) * rate
    heldout["eur_to_gbp"] = rate
    if history_metadata:
        heldout["preseason_valuation_date"] = test.preseason_valuation_date
        heldout["preseason_value_gbp"] = test.preseason_value_million_eur * 1_000_000 * rate
        heldout["preseason_persistence_prediction_gbp"] = persistence
    data["split"] = np.where(data.valuation_date < test.valuation_date.min(), "train", "test")
    atomic_write(data_dir / "training_dataset.csv", data.to_csv(index=False))
    atomic_write(data_dir / "holdout_predictions.csv", heldout.to_csv(index=False))
    coefficients = pd.DataFrame({"feature": feature_names, "coefficient_gbp": regressor.coef_ * rate,
                                 "eur_to_gbp": rate})
    atomic_write(model_dir / "coefficients.csv", coefficients.to_csv(index=False))
    atomic_write(model_dir / "metrics.json", json.dumps(report, indent=2, allow_nan=False) + "\n")
    artifact = {"schema_version": 1, "pipeline": pipeline, "metadata": report}
    joblib.dump(artifact, model_dir / "model.joblib", compress=3)
    from .plots import evaluation_plot
    plot = evaluation_plot(heldout, report, plot_dir / "evaluation.png")
    return {"model_path": model_dir / "model.joblib", "report_path": model_dir / "metrics.json",
            "dataset_path": data_dir / "training_dataset.csv", "plot_path": plot, "report": report}
