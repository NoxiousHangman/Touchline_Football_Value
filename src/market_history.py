"""Strictly preseason Transfermarkt values, never current or target valuations."""

import pandas as pd

from .features import dates, numbers, require_columns

HISTORY_FEATURES = ["preseason_value_million_eur", "preseason_value_missing"]
HISTORY_MAX_AGE_DAYS = 365


def add_market_history(frame: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Use the latest value strictly before window_start, no more than 365 days old.

    Input frame must already have passed normal feature/dataset validation.
    History uses the published Transfermarkt schema and its ID namespace.
    Missing history stays missing, with an explicit missingness feature.
    """
    if not frame.player_id_namespace.eq("transfermarkt").all() or frame.is_synthetic.any():
        raise ValueError("Market history requires real observations with Transfermarkt player IDs.")
    require_columns(history, ["player_id", "date", "market_value_in_eur"], "Market history")
    right = history[["player_id", "date", "market_value_in_eur"]].copy()
    right["player_id"] = numbers(right.player_id, "history player_id", nullable=False, integer=True).astype("int64")
    if right.player_id.eq(0).any():
        raise ValueError("History player_id must be positive.")
    right["preseason_valuation_date"] = dates(right.pop("date"), "history date")
    right["preseason_value_million_eur"] = numbers(
        right.pop("market_value_in_eur"), "history market value", nullable=False) / 1_000_000
    if right.duplicated(["player_id", "preseason_valuation_date"]).any():
        raise ValueError("Market history contains duplicate player/date rows.")
    # Discard supplied derived columns so stale or edited values cannot bypass the join.
    left = frame.drop(columns=HISTORY_FEATURES + ["preseason_valuation_date"], errors="ignore").copy()
    left["_history_order"] = range(len(left))
    left["_season_start"] = dates(left.window_start, "window_start")
    result = pd.merge_asof(
        left.sort_values("_season_start"), right.sort_values("preseason_valuation_date"),
        left_on="_season_start", right_on="preseason_valuation_date", by="player_id",
        direction="backward", allow_exact_matches=False,
        tolerance=pd.Timedelta(HISTORY_MAX_AGE_DAYS, unit="D"),
    ).sort_values("_history_order")
    result["preseason_value_missing"] = result.preseason_value_million_eur.isna().astype(int)
    result["preseason_valuation_date"] = result.preseason_valuation_date.dt.strftime("%Y-%m-%d")
    result = result.drop(columns=["_season_start", "_history_order"])
    result.index = frame.index
    return result
