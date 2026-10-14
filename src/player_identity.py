"""Exact published name aliases plus exact birthdays; never fuzzy identity joins."""
from collections import Counter, defaultdict
import unicodedata

import pandas as pd


def normalise(value):
    return "".join(c for c in unicodedata.normalize("NFKD", str(value).casefold()) if c.isalnum())


def match_profiles(records, players):
    """Return source-ID -> TM-ID only when both directions are unambiguous."""
    index = defaultdict(set)
    valid = players.loc[pd.to_datetime(players.date_of_birth, errors="coerce").notna()]
    for row in valid.to_dict("records"):
        dob = str(row.get("date_of_birth", ""))[:10]
        names = [row.get("name")]
        if pd.notna(row.get("first_name")) and pd.notna(row.get("last_name")):
            names.append(f"{row['first_name']} {row['last_name']}")
        for name in names:
            if isinstance(name, str) and name.strip():
                index[(normalise(name), dob)].add(int(row["player_id"]))
    ids = Counter(str(r["id"]) for r in records)
    matches = {}
    for record in records:
        source_id = str(record["id"])
        if ids[source_id] != 1:
            continue
        dob = str(record.get("dob", ""))[:10]
        candidates = set()
        for name in [record["name"], *record.get("aliases", [])]:
            if isinstance(name, str) and name.strip():
                candidates.update(index.get((normalise(name), dob), set()))
        if len(candidates) == 1:
            matches[source_id] = candidates.pop()
    counts = Counter(matches.values())
    return {source: target for source, target in matches.items() if counts[target] == 1}
