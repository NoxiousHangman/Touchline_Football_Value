"""Use the published dcaribou Transfermarkt snapshot for historical EUR labels."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unicodedata

import pandas as pd
import requests

from .api_client import atomic_write
from .features import boolean_column, dates, numbers, require_columns, validate_features

DATASET_PAGE = "https://github.com/dcaribou/transfermarkt-datasets"
DOWNLOAD_BASE = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"
VALUE_SOURCE = "Transfermarkt via dcaribou/transfermarkt-datasets"
MAPPING_COLUMNS = ["football_data_player_id", "transfermarkt_player_id", "verified", "evidence"]


def download_snapshot(output_dir: Path, include_statistics: bool = False) -> dict:
    """Download published files, without scraping Transfermarkt or using API secrets."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = ["players", "player_valuations"] + (["games", "appearances"] if include_statistics else [])
    paths = [output_dir / f"{table}.csv.gz" for table in tables]
    metadata = {"dataset": DATASET_PAGE, "downloaded_at": datetime.now(timezone.utc).isoformat(),
                "publisher_license": "CC0 (publisher dataset metadata)", "files": {}}
    metadata_path = output_dir / "download_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for path in paths:
        if path.exists():
            recorded = metadata["files"].get(path.name, {})
            if hashlib.sha256(path.read_bytes()).hexdigest() != recorded.get("sha256"):
                raise ValueError(f"Existing {path.name} does not match recorded provenance. Use a new directory.")
            continue
        temporary = None
        try:
            digest, size = hashlib.sha256(), 0
            with requests.get(f"{DOWNLOAD_BASE}/{path.name}", stream=True, timeout=(10, 60)) as response:
                response.raise_for_status()
                with tempfile.NamedTemporaryFile(dir=output_dir, suffix=".csv.gz", delete=False) as handle:
                    temporary = Path(handle.name)
                    for chunk in response.iter_content(1024 * 1024):
                        size += len(chunk)
                        if size > 100 * 1024 * 1024:
                            raise ValueError("Unexpected download larger than 100 MB; stopped.")
                        digest.update(chunk)
                        handle.write(chunk)
            expected = {"players.csv.gz": ["player_id", "name", "date_of_birth"],
                        "player_valuations.csv.gz": ["player_id", "date", "market_value_in_eur"],
                        "games.csv.gz": ["game_id", "competition_id", "season", "date"],
                        "appearances.csv.gz": ["game_id", "player_id", "minutes_played", "goals"]}[path.name]
            require_columns(pd.read_csv(temporary, nrows=2), expected, path.name)
            temporary.replace(path)
            metadata["files"][path.name] = {"url": f"{DOWNLOAD_BASE}/{path.name}", "bytes": size,
                                           "sha256": digest.hexdigest(), "downloaded_at": datetime.now(timezone.utc).isoformat()}
            atomic_write(output_dir / "download_metadata.json", json.dumps(metadata, indent=2) + "\n")
        except requests.RequestException:
            raise ValueError("Transfermarkt dataset download failed. Check the connection or use downloaded CSV files.") from None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return metadata


def normalize_name(name: str) -> str:
    decomposed = unicodedata.normalize("NFKD", str(name)).casefold()
    return "".join(char for char in decomposed if char.isalnum())


def mapping_candidates(features: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    observations = validate_features(features)
    if observations.player_id_namespace.iloc[0] != "football-data.org":
        raise ValueError("ID mapping is only for football-data.org observations; Transfermarkt statistics already share IDs.")
    require_columns(players, ["player_id", "name", "date_of_birth"], "Transfermarkt players")
    people = observations[["player_id", "player_name", "date_of_birth"]].drop_duplicates()
    if people.player_id.duplicated().any():
        raise ValueError("Inconsistent names/birth dates for a football-data.org ID; reconcile before mapping.")
    people = people.rename(columns={"player_id": "football_data_player_id"})
    people["name_key"] = people.player_name.map(normalize_name)
    other = players[["player_id", "name", "date_of_birth"]].copy().rename(
        columns={"player_id": "transfermarkt_player_id", "name": "transfermarkt_name"})
    other["name_key"] = other.transfermarkt_name.map(normalize_name)
    # Dataset dates may include a midnight timestamp.
    other["date_of_birth"] = other.date_of_birth.astype(str).str[:10]
    candidates = people.merge(other, on=["name_key", "date_of_birth"], how="left").drop(columns="name_key")
    candidates["transfermarkt_player_id"] = candidates.transfermarkt_player_id.astype("Int64")
    candidates["verified"] = False
    candidates["evidence"] = ""
    candidates["match_basis"] = candidates.transfermarkt_player_id.notna().map(
        {True: "normalized name + birth date; verify identity", False: "no exact candidate; manual review needed"})
    return candidates


def historical_labels(features: pd.DataFrame, valuations: pd.DataFrame, mapping: pd.DataFrame,
                       max_delay_days: int = 180) -> tuple[pd.DataFrame, dict]:
    """Choose the first available label strictly after each season observation."""
    if max_delay_days < 1:
        raise ValueError("max_delay_days must be positive.")
    observations = validate_features(features)
    if observations.player_id_namespace.iloc[0] != "football-data.org":
        raise ValueError("This cross-source importer requires football-data.org observations.")
    if observations.is_synthetic.any():
        raise ValueError("Transfermarkt labels require real football-data.org observations.")
    require_columns(mapping, MAPPING_COLUMNS, "Player mapping")
    mapping = mapping[MAPPING_COLUMNS].copy()
    if not boolean_column(mapping.verified, "verified").all():
        raise ValueError("Every supplied mapping must be reviewed and marked verified=true; candidates are not confirmed IDs.")
    for col in MAPPING_COLUMNS[:2]:
        mapping[col] = numbers(mapping[col], col, nullable=False, integer=True).astype("int64")
        if (mapping[col] <= 0).any() or mapping[col].duplicated().any():
            raise ValueError("Player ID mapping must be positive and one-to-one in both directions.")
    if mapping.evidence.isna().any() or mapping.evidence.astype(str).str.strip().eq("").any():
        raise ValueError("Each verified player mapping needs identity evidence (e.g. profile URLs and birth date).")
    require_columns(valuations, ["player_id", "date", "market_value_in_eur"], "Transfermarkt valuations")
    raw = valuations[["player_id", "date", "market_value_in_eur"]].rename(
        columns={"player_id": "transfermarkt_player_id", "date": "valuation_date", "market_value_in_eur": "market_value_eur"})
    raw["transfermarkt_player_id"] = numbers(raw.transfermarkt_player_id, "Transfermarkt player_id",
                                            nullable=False, integer=True).astype("int64")
    raw = raw.merge(mapping, on="transfermarkt_player_id", how="inner", validate="many_to_one")
    if raw.empty:
        raise ValueError("No Transfermarkt records matched the reviewed ID mapping.")
    raw["valuation_date"] = dates(raw.valuation_date.astype(str).str[:10], "valuation_date")
    raw["market_value_eur"] = numbers(raw.market_value_eur, "market_value_eur", nullable=False)
    raw = raw.drop_duplicates(["football_data_player_id", "valuation_date", "market_value_eur"])
    if raw.duplicated(["football_data_player_id", "valuation_date"]).any():
        raise ValueError("Conflicting Transfermarkt values on the same player/date; resolve before importing.")
    raw = raw.rename(columns={"football_data_player_id": "player_id"})
    left = observations[["player_id", "observation_date"]].copy()
    left["observation_date"] = dates(left.observation_date, "observation_date")
    joined = pd.merge_asof(left.sort_values("observation_date"), raw.sort_values("valuation_date"),
                           by="player_id", left_on="observation_date", right_on="valuation_date",
                           direction="forward", allow_exact_matches=False,
                           tolerance=pd.Timedelta(max_delay_days, unit="D"))
    matched = joined.valuation_date.notna()
    labels = joined.loc[matched, ["player_id", "valuation_date", "market_value_eur"]].drop_duplicates()
    if labels.empty:
        raise ValueError("No valuations fall after these observations within the chosen label window.")
    labels["valuation_date"] = labels.valuation_date.dt.strftime("%Y-%m-%d")
    labels["value_source"] = VALUE_SOURCE
    labels["is_synthetic"] = False
    report = {"source": DATASET_PAGE, "selection": "first valuation strictly after observation",
              "max_delay_days": max_delay_days, "observations": len(left), "matched_observations": int(matched.sum()),
              "unmatched_observations": int((~matched).sum()),
              "unmatched_player_ids": sorted(joined.loc[~matched, "player_id"].unique().tolist()),
              "valuation_rows": len(labels)}
    return labels, report
