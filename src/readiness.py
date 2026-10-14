"""Read-only project status; never load model binaries or expose credentials."""

import json
import os
from pathlib import Path

from .config import MODELS_DIR, RAW_DIR, TOKEN_ENV_VAR


def project_status(raw_dir: Path = RAW_DIR, models_dir: Path = MODELS_DIR) -> dict:
    snapshots = {name: (Path(raw_dir) / "transfermarkt" / f"{name}.csv.gz").is_file()
                 for name in ("players", "player_valuations", "games", "appearances")}
    runs = []
    for path in sorted(Path(models_dir).glob("*/metrics.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(report.get("is_synthetic"), bool):
                continue
            runs.append({"name": path.parent.name, "synthetic": report["is_synthetic"],
                         "model_exists": (path.parent / "model.joblib").is_file(),
                         "currency": report.get("currency"), "source": report.get("player_id_namespace", "football-data.org"),
                         "train_rows": report.get("train_rows"), "test_rows": report.get("test_rows")})
        except (OSError, ValueError, AttributeError):
            continue
    real = [run for run in runs if not run["synthetic"] and run["model_exists"]]
    return {"football_data_token_configured": bool(os.environ.get(TOKEN_ENV_VAR, "").strip()),
            "transfermarkt_files": snapshots, "model_runs": runs, "real_model_available": bool(real),
            "next_step": "Use the real model with matching-source player features." if real else
                         "Prepare Transfermarkt statistics and train." if all(snapshots.values()) else
                         "Download Transfermarkt statistics or configure football-data.org access."}
