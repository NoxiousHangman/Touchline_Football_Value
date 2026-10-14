"""Reproducible GBP presentation of the source dataset's EUR values."""

import math

DEFAULT_EUR_TO_GBP = 0.85950
FX_REFERENCE_DATE = "2026-09-23"
FX_SOURCE = "https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/eurofxref-graph-gbp.en.html"


def gbp_conversion(rate: float | None = None) -> dict:
    """A fixed presentation rate, not historical per-valuation FX or a live feed."""
    value = DEFAULT_EUR_TO_GBP if rate is None else float(rate)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("EUR-to-GBP rate must be a finite positive number (pounds per euro).")
    return {"eur_to_gbp": value, "reference_date": FX_REFERENCE_DATE if rate is None else None,
            "source": FX_SOURCE if rate is None else "user-supplied rate",
            "method": "fixed presentation conversion; GBP = EUR * rate"}
