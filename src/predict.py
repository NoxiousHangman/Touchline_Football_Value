"""Apply a locally trained pipeline and expose additive linear contributions."""

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn

from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, validate_features
from .currency import gbp_conversion
from .market_history import add_market_history, HISTORY_FEATURES


def load_model(path: Path) -> dict:
    """Load only trusted models created by this project: joblib can execute code."""
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or artifact.get("schema_version") != 1:
        raise ValueError("Unsupported model artifact. Retrain using this project.")
    metadata = artifact["metadata"]
    expected_numeric = NUMERIC_FEATURES + (HISTORY_FEATURES if metadata.get("market_history") else [])
    if (metadata["numeric_features"] != expected_numeric or metadata["categorical_features"] != CATEGORICAL_FEATURES):
        raise ValueError("Model feature schema has changed. Retrain before predicting.")
    if metadata["versions"]["sklearn"] != sklearn.__version__:
        raise ValueError("scikit-learn version differs from training. Use the recorded version or retrain.")
    return artifact


def predict_players(features: pd.DataFrame, artifact: dict, player_id: int | None = None,
                    eur_to_gbp: float | None = None,
                    valuation_history: pd.DataFrame | None = None) -> pd.DataFrame:
    # Existing EUR models work too; never multiply their stored coefficients in place.
    fx = (gbp_conversion(eur_to_gbp) if eur_to_gbp is not None
          else artifact["metadata"].get("fx", gbp_conversion()))
    rate = gbp_conversion(fx["eur_to_gbp"])["eur_to_gbp"]
    frame = validate_features(features)
    if frame.player_id_namespace.iloc[0] != artifact["metadata"].get("player_id_namespace", "football-data.org"):
        raise ValueError("Prediction player ID namespace differs from the model's data source.")
    if bool(frame.is_synthetic.iloc[0]) != artifact["metadata"]["is_synthetic"]:
        raise ValueError("Do not use a synthetic demo model on real players (or the reverse).")
    if artifact["metadata"].get("market_history"):
        if valuation_history is None:
            raise ValueError("This model requires --valuation-history with dated Transfermarkt values.")
        frame = add_market_history(frame, valuation_history)
    numeric_features = artifact["metadata"]["numeric_features"]
    model_features = numeric_features + CATEGORICAL_FEATURES
    # Compute history before selecting a player/date.
    if player_id is not None:
        frame = frame.loc[frame.player_id == player_id].tail(1)
        if frame.empty:
            raise ValueError(f"Player {player_id} does not appear in the supplied features.")
    pipeline = artifact["pipeline"]
    preprocess = pipeline.named_steps["preprocess"]
    regressor = pipeline.named_steps["regression"]
    # Unknown categories get a per-row explanation below, instead of a generic library warning.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Found unknown categories.*")
        transformed = preprocess.transform(frame[model_features])
    contributions = transformed * regressor.coef_ * rate
    intercept = regressor.intercept_ * rate
    raw = contributions.sum(axis=1) + intercept
    result = frame[["player_id", "player_name", "observation_date", "is_synthetic"]].copy().reset_index(drop=True)
    result["raw_prediction_gbp"] = raw
    result["predicted_market_value_gbp"] = np.maximum(0, raw)
    result["intercept_gbp"] = intercept
    result["zero_floor_adjustment_gbp"] = np.maximum(0, raw) - raw
    result["currency"] = "GBP"
    result["player_id_namespace"] = frame.player_id_namespace.iloc[0]
    result["eur_to_gbp"] = rate
    result["fx_reference_date"] = fx["reference_date"]
    result["fx_source"] = fx["source"]
    result["missing_numeric_features"] = frame[numeric_features].isna().sum(axis=1).to_numpy()
    if artifact["metadata"].get("market_history"):
        result["preseason_valuation_date"] = frame.preseason_valuation_date.to_numpy()
        result["preseason_value_gbp"] = frame.preseason_value_million_eur.to_numpy() * 1_000_000 * rate
        result["preseason_value_missing"] = frame.preseason_value_missing.to_numpy()
    categories = preprocess.named_transformers_["category"].categories_
    result["unseen_categories"] = [
        "; ".join(f"{col}={row[col]}" for col, known in zip(CATEGORICAL_FEATURES, categories) if row[col] not in known)
        for _, row in frame.iterrows()
    ]
    result["predates_training_cutoff"] = result.observation_date <= artifact["metadata"]["train_date_max"]
    for index, name in enumerate(preprocess.get_feature_names_out()):
        result[f"contribution_{name}_gbp"] = contributions[:, index]
    return result
