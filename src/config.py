"""Paths are relative to this project, independent of the shell's directory."""

from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("TOUCHLINE_DATA_DIR", str(PROJECT_ROOT / "data")))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = Path(os.environ.get("TOUCHLINE_MODELS_DIR", str(PROJECT_ROOT / "models")))
VISUALISATIONS_DIR = PROJECT_ROOT / "visualisations"
API_BASE_URL = "https://api.football-data.org/v4"
TOKEN_ENV_VAR = "FOOTBALL_DATA_API_TOKEN"
ATTRIBUTION = "Football data provided by the Football-Data.org API"
