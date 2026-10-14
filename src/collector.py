"""Flatten API responses for inspection, keeping raw responses as the source of truth."""

import json
from pathlib import Path

import pandas as pd

from .api_client import FetchResult, FootballDataError, atomic_write
from .config import ATTRIBUTION, PROCESSED_DIR


def export_csv(result: FetchResult, resource: str, output_dir: Path = PROCESSED_DIR) -> list[Path]:
    payload = result.data
    key = {"scorers": "scorers", "matches": "matches", "player-matches": "matches"}.get(resource)
    if resource == "player":
        if "id" not in payload:
            raise FootballDataError("Player response has no ID. Raw JSON is retained for inspection.")
        records = [payload]
    elif key:
        records = payload.get(key)
        if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
            raise FootballDataError(f"Response has no valid {key} list. Raw JSON is retained for inspection.")
    else:
        raise ValueError(f"Unsupported resource: {resource}")

    frame = pd.json_normalize(records, sep="_")
    if frame.empty and len(frame.columns) == 0:
        frame = pd.DataFrame(columns=["player_id" if resource == "scorers" else "id"])
    # Preserve response-level context for later season joins.
    for field in ("competition", "season", "filters", "person"):
        value = payload.get(field)
        if isinstance(value, dict):
            flattened = pd.json_normalize([value], sep="_").to_dict("records")[0]
            for column, cell in flattened.items():
                frame[f"context_{field}_{column}"] = [cell] * len(frame)
    frame["source_fetched_at"] = result.fetched_at
    frame["source_cache_file"] = result.cache_path.name
    frame["source_attribution"] = ATTRIBUTION
    frames = [(resource, frame)]
    aggregate = payload.get("aggregations")
    if resource == "player-matches" and isinstance(aggregate, dict):
        aggregate_frame = pd.json_normalize([aggregate], sep="_")
        aggregate_frame["scope"] = "returned_matches_only"
        aggregate_frame["player_id"] = (payload.get("person") or {}).get("id")
        aggregate_frame["source_cache_file"] = result.cache_path.name
        aggregate_frame["source_fetched_at"] = result.fetched_at
        aggregate_frame["source_attribution"] = ATTRIBUTION
        frames.append(("aggregates", aggregate_frame))

    paths = []
    for label, table in frames:
        # Lists/dicts remain JSON cells rather than ambiguous Python repr strings.
        for column in table.columns:
            table[column] = table[column].map(
                lambda value: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
            )
        path = Path(output_dir) / f"{result.cache_path.stem}_{label}.csv"
        atomic_write(path, table.to_csv(index=False, lineterminator="\n"))
        paths.append(path)
    return paths
