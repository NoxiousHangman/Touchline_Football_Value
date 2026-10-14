"""Validate dated player snapshots and derive model features without targets."""

import numpy as np
import pandas as pd

COUNTS = ["appearances", "starts", "minutes_played", "goals", "assists", "yellow_cards", "red_cards"]
IDENTITY = ["player_id", "player_name", "date_of_birth", "position", "competition_code",
            "season_start_year", "window_start", "window_end", "observation_date",
            "coverage_complete", "source", "is_synthetic"]
OBSERVATION_COLUMNS = IDENTITY + COUNTS
NUMERIC_FEATURES = ["age", "minutes_played", "starting_fraction", "goals_per90", "assists_per90",
                    "yellow_cards_per90", "red_cards_per90"] + [
    f"previous_{lag}_{field}" for lag in (1, 2)
    for field in ("minutes_played", "goals_per90", "assists_per90")
]
CATEGORICAL_FEATURES = ["position", "competition_code"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError(f"{label} contains no rows.")


def boolean_column(series: pd.Series, name: str) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower().map({"true": True, "false": False})
    if normalized.isna().any():
        raise ValueError(f"{name} must explicitly contain true or false on every row.")
    return normalized.astype(bool)


def dates(series: pd.Series, name: str) -> pd.Series:
    text = series.astype(str)
    if not text.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
        raise ValueError(f"{name} must use YYYY-MM-DD dates without times.")
    result = pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")
    if result.isna().any():
        raise ValueError(f"{name} contains missing or invalid dates.")
    return result


def numbers(series: pd.Series, name: str, *, nullable: bool = True, integer: bool = False) -> pd.Series:
    result = pd.to_numeric(series, errors="coerce")
    bad = (series.notna() & result.isna()) | np.isinf(result) | (result < 0)
    if not nullable:
        bad |= result.isna()
    if integer:
        bad |= result.notna() & (result % 1 != 0)
    if bad.any():
        raise ValueError(f"{name} must contain finite non-negative {'integers' if integer else 'numbers'}"
                         + (" or blank values." if nullable else " on every row."))
    return result


def validate_observations(observations: pd.DataFrame) -> pd.DataFrame:
    require_columns(observations, OBSERVATION_COLUMNS, "Observations")
    frame = observations[OBSERVATION_COLUMNS].copy().reset_index(drop=True)
    frame["player_id_namespace"] = (observations["player_id_namespace"].to_numpy()
                                    if "player_id_namespace" in observations else "football-data.org")
    if (frame.player_id_namespace.nunique(dropna=False) != 1
            or not frame.player_id_namespace.isin(["football-data.org", "transfermarkt"]).all()):
        raise ValueError("Use exactly one player_id_namespace: football-data.org or transfermarkt.")
    frame["player_id"] = numbers(frame.player_id, "player_id", nullable=False, integer=True).astype("int64")
    if (frame.player_id == 0).any():
        raise ValueError("player_id must be positive and match the declared player_id_namespace.")
    frame["season_start_year"] = numbers(frame.season_start_year, "season_start_year", nullable=False,
                                          integer=True).astype("int64")
    frame["is_synthetic"] = boolean_column(frame.is_synthetic, "is_synthetic")
    frame["coverage_complete"] = boolean_column(frame.coverage_complete, "coverage_complete")
    if not frame.coverage_complete.all():
        raise ValueError("Incomplete windows cannot enter the season model. Review coverage_complete first.")
    if frame.is_synthetic.nunique() != 1:
        raise ValueError("Do not mix synthetic and real observations.")
    for col in ("player_name", "position", "competition_code", "source"):
        if frame[col].isna().any() or frame[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"{col} is required on every observation.")
        frame[col] = frame[col].astype(str).str.strip()
    parsed = {col: dates(frame[col], col) for col in
              ("date_of_birth", "window_start", "window_end", "observation_date")}
    if ((parsed["window_start"] > parsed["window_end"])
            | (parsed["window_end"] > parsed["observation_date"])).any():
        raise ValueError("Require window_start <= window_end <= observation_date.")
    if not parsed["window_start"].dt.year.eq(frame.season_start_year).all():
        raise ValueError("season_start_year must equal the year of window_start.")
    duration = (parsed["window_end"] - parsed["window_start"]).dt.days
    if (duration > 400).any():
        raise ValueError("A season window must not exceed 400 days.")
    age = (parsed["observation_date"] - parsed["date_of_birth"]).dt.days / 365.2425
    if ((age < 14) | (age > 60)).any():
        raise ValueError("Age at observation must be between 14 and 60; check dates of birth.")
    if frame.duplicated(["player_id", "season_start_year"]).any():
        raise ValueError("Use one reviewed observation per player and season, including transferred players.")
    ordered = frame.sort_values(["player_id", "window_start"])
    previous_end = ordered.groupby("player_id").window_end.shift()
    if (ordered.window_start <= previous_end).any():
        raise ValueError("A player's season windows must not overlap.")
    for col in COUNTS:
        frame[col] = numbers(frame[col], col, integer=True)
    if (frame.starts > frame.appearances).any():
        raise ValueError("starts cannot exceed appearances.")
    if ((frame.appearances == 0) & (frame[COUNTS[1:]].fillna(0) > 0).any(axis=1)).any():
        raise ValueError("A player with zero appearances cannot have positive match statistics.")
    frame["age"] = age
    return frame


def build_features(observations: pd.DataFrame) -> pd.DataFrame:
    frame = validate_observations(observations).sort_values(["player_id", "season_start_year"])
    minutes = frame.minutes_played.where(frame.minutes_played > 0)
    appearances = frame.appearances.where(frame.appearances > 0)
    for col in ("goals", "assists", "yellow_cards", "red_cards"):
        frame[f"{col}_per90"] = 90 * frame[col] / minutes
    frame["goal_contributions"] = frame.goals + frame.assists
    frame["contributions_per90"] = 90 * frame.goal_contributions / minutes
    frame["starting_fraction"] = frame.starts / appearances
    frame["minutes_per_appearance"] = frame.minutes_played / appearances
    # Match exact season offsets, not simply the previous row (which may skip years).
    history_fields = ["minutes_played", "goals_per90", "assists_per90"]
    history = frame[["player_id", "season_start_year", "observation_date", "window_end"] + history_fields]
    for lag in (1, 2):
        prior = history.copy()
        prior["season_start_year"] += lag
        rename = {col: f"previous_{lag}_{col}" for col in history_fields + ["observation_date", "window_end"]}
        frame = frame.merge(prior.rename(columns=rename), on=["player_id", "season_start_year"],
                            how="left", validate="one_to_one")
        usable = ((frame[f"previous_{lag}_observation_date"] < frame.observation_date)
                  & (frame[f"previous_{lag}_window_end"] < frame.window_start))
        frame.loc[~usable, [f"previous_{lag}_{col}" for col in history_fields]] = np.nan
        frame = frame.drop(columns=[f"previous_{lag}_observation_date", f"previous_{lag}_window_end"])
    return frame.sort_values(["observation_date", "player_id"]).reset_index(drop=True)


def validate_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Recompute engineered values so edited/stale derived CSV cells are not trusted."""
    return build_features(frame)
