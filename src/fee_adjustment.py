"""Transparent user-requested fee scenario, separate from trained market values."""
import json
import math
from statistics import median

from .config import PROJECT_ROOT

REFERENCE = json.loads((PROJECT_ROOT / "docs/transfer_fee_reference.json").read_text(encoding="utf-8"))
FEE_MULTIPLIER = median(row["fee"] / row["market_value"] for row in REFERENCE["rows"])


def adjustment_info():
    return {"kind": "illustrative_fee_scenario", "multiplier": FEE_MULTIPLIER,
            "uplift_percent": (FEE_MULTIPLIER - 1) * 100, "sample_rows": len(REFERENCE["rows"]),
            "reference_currency": REFERENCE["currency"], "output_currency": "GBP",
            "source": "User-provided transfer table", "method": REFERENCE["method"],
            "limitation": REFERENCE["limitation"], "validated": False}


def with_fee_scenario(record):
    """Return a new record; never compound an earlier adjustment or alter the base."""
    base = record.get("predicted_market_value_gbp")
    reference = record.get("valuation_basis") == "preseason_reference"
    if reference:
        base = record.get("reference_market_value_gbp")
    usable = (record.get("estimate_available", True) or reference) and isinstance(base, (float, int)) and math.isfinite(base) and base >= 0
    return dict(record, illustrative_transfer_fee_gbp=base * FEE_MULTIPLIER if usable else None,
                transfer_fee_uplift_percent=(FEE_MULTIPLIER - 1) * 100)
