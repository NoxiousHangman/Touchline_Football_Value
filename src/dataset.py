"""Join dated market-value labels to strictly earlier player observations."""

import pandas as pd

from .features import boolean_column, dates, numbers, require_columns, validate_features

VALUATION_COLUMNS = ["player_id", "valuation_date", "market_value_eur", "value_source", "is_synthetic"]


def build_training_dataset(features: pd.DataFrame, valuations: pd.DataFrame,
                           max_age_days: int = 180) -> pd.DataFrame:
    if max_age_days < 1:
        raise ValueError("max_age_days must be positive.")
    stats = validate_features(features)
    require_columns(valuations, VALUATION_COLUMNS, "Valuations")
    # Only explicitly permitted label columns enter the join.
    labels = valuations[VALUATION_COLUMNS].copy()
    namespace = valuations.get("player_id_namespace", pd.Series("football-data.org", index=valuations.index))
    if not namespace.eq(stats.player_id_namespace.iloc[0]).all():
        raise ValueError("Valuation and observation player ID namespaces differ; use a reviewed mapping.")
    labels["player_id"] = numbers(labels.player_id, "player_id", nullable=False, integer=True).astype("int64")
    labels["market_value_eur"] = numbers(labels.market_value_eur, "market_value_eur", nullable=False)
    labels["valuation_date"] = dates(labels.valuation_date, "valuation_date")
    labels["is_synthetic"] = boolean_column(labels.is_synthetic, "is_synthetic")
    if labels.is_synthetic.nunique() != 1 or labels.is_synthetic.iloc[0] != stats.is_synthetic.iloc[0]:
        raise ValueError("Valuations and observations must have matching real/synthetic provenance.")
    if labels.value_source.isna().any() or labels.value_source.astype(str).str.strip().eq("").any():
        raise ValueError("value_source is required for every valuation.")
    if labels.duplicated(["player_id", "valuation_date"]).any():
        raise ValueError("Duplicate player/date valuations: resolve conflicting labels before training.")
    # Strip label-like columns even if an input feature file was previously joined.
    stats = stats.drop(columns=[col for col in VALUATION_COLUMNS[1:-1] if col in stats])
    stats["observation_date"] = dates(stats.observation_date, "observation_date")
    labels = labels.drop(columns="is_synthetic")
    joined = pd.merge_asof(labels.sort_values("valuation_date"), stats.sort_values("observation_date"),
                           by="player_id", left_on="valuation_date", right_on="observation_date",
                           direction="backward", allow_exact_matches=False,
                           tolerance=pd.Timedelta(max_age_days, unit="D"))
    unmatched = joined.observation_date.isna()
    if unmatched.any():
        sample = joined.loc[unmatched, "player_id"].head(5).tolist()
        raise ValueError(f"{unmatched.sum()} valuation(s) have no strictly earlier observation within "
                         f"{max_age_days} days (example IDs: {sample}). No labels were silently dropped.")
    joined["feature_age_days"] = (joined.valuation_date - joined.observation_date).dt.days
    for col in ("observation_date", "valuation_date"):
        joined[col] = joined[col].dt.strftime("%Y-%m-%d")
    return joined.sort_values(["valuation_date", "player_id"]).reset_index(drop=True)
