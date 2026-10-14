"""Deterministic invented data for exercising software, never real valuations."""

import numpy as np
import pandas as pd

from .features import OBSERVATION_COLUMNS


def demo_data(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    observations, values = [], []
    positions = ["Goalkeeper", "Defence", "Midfield", "Attack"]
    for player_id in range(1, 81):
        birth_year = int(rng.integers(1989, 2005))
        position = positions[player_id % 4]
        elite = player_id <= 6
        if elite:
            birth_year, position = 2000, "Attack"
        league = ["DEMO_A", "DEMO_B"][player_id % 2]
        for year in (2023, 2024, 2025):
            appearances = int(rng.integers(8, 39))
            starts = int(rng.integers(1, appearances + 1))
            minutes = starts * 76 + (appearances - starts) * 22
            goals = int(rng.poisson(minutes / 90 * (.45 if position == "Attack" else .09)))
            assists = int(rng.poisson(minutes / 90 * .14))
            if elite:
                appearances, starts = 38, 36
                minutes = int(rng.integers(3100, 3301))
                goals = int(rng.integers(46, 57))
                assists = int(rng.integers(10, 20))
            row = dict(player_id=player_id, player_name=f"Demo Player {player_id:03}",
                       date_of_birth=f"{birth_year}-03-15", position=position, competition_code=league,
                       season_start_year=year, window_start=f"{year}-07-01", window_end=f"{year+1}-05-31",
                       observation_date=f"{year+1}-06-01", coverage_complete=True,
                       source="SYNTHETIC software demo; no real player data", is_synthetic=True,
                       appearances=appearances, starts=starts, minutes_played=minutes, goals=goals,
                       assists=assists, yellow_cards=int(rng.integers(0, 9)), red_cards=int(rng.integers(0, 3)))
            observations.append(row)
            age = year + 1 - birth_year
            # Invented high-end examples exercise £200m+ outputs; never rescale real labels.
            value = (38e6 - age * .9e6 + minutes * 8500 + goals / minutes * 90 * 140e6
                     + assists / minutes * 90 * 12e6 + (3e6 if league == "DEMO_A" else 0)
                     + rng.normal(0, 3e6))
            values.append(dict(player_id=player_id, valuation_date=f"{year+1}-06-15",
                               market_value_eur=round(max(100000, value)), value_source="SYNTHETIC demo",
                               is_synthetic=True))
    return pd.DataFrame(observations)[OBSERVATION_COLUMNS], pd.DataFrame(values)
