"""Apply inclusive budget/age filters before pagination, in the displayed currency."""
import math

from .fee_adjustment import with_fee_scenario


def parse_filters(query, live=False):
    result = {}
    for key in ("min_value", "max_value", "min_age", "max_age"):
        raw = query.get(key, [""])[0]
        if raw == "":
            result[key] = None
            continue
        try:
            number = float(raw)
        except ValueError:
            raise ValueError("Enter valid numeric range limits.") from None
        if not math.isfinite(number) or number < 0 or (key.endswith("age") and number > 120):
            raise ValueError("Range limits must be finite, non-negative numbers.")
        result[key] = number
    for low, high in (("min_value", "max_value"), ("min_age", "max_age")):
        if result[low] is not None and result[high] is not None and result[low] > result[high]:
            raise ValueError("The minimum cannot exceed the maximum.")
    result["value_view"] = query.get("value_view", ["fee" if live else "market"])[0]
    result["position"] = query.get("position", [""])[0]
    result["sort"] = query.get("sort", ["value_desc"])[0]
    if result["value_view"] not in ("fee", "market") or (not live and result["value_view"] != "market"):
        raise ValueError("Invalid value view.")
    if result["position"] not in ("", "Goalkeeper", "Defender", "Midfield", "Attack"):
        raise ValueError("Invalid position.")
    if result["sort"] not in ("value_desc", "value_asc", "age_asc", "name"):
        raise ValueError("Invalid sort order.")
    return result


def position_group(value):
    value = str(value).casefold()
    if "goalkeep" in value:
        return "Goalkeeper"
    if "defend" in value or "back" in value:
        return "Defender"
    if "midfield" in value:
        return "Midfield"
    if any(word in value for word in ("attack", "forward", "striker", "winger")):
        return "Attack"
    return None


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def filter_players(rows, filters):
    def price(row):
        if filters["value_view"] == "fee":
            return with_fee_scenario(row)["illustrative_transfer_fee_gbp"]
        return row.get("valuation_value_gbp", row.get("predicted_market_value_gbp"))

    def within(value, low, high):
        if low is None and high is None:
            return True
        return finite(value) and (low is None or value >= low) and (high is None or value <= high)

    selected = []
    for row in rows:
        age = math.floor(row["age"]) if finite(row.get("age")) else None
        if not within(price(row), filters["min_value"], filters["max_value"]):
            continue
        if not within(age, filters["min_age"], filters["max_age"]):
            continue
        if filters["position"] and position_group(row.get("position")) != filters["position"]:
            continue
        selected.append(row)

    def key(row):
        name = str(row["player_name"]).casefold()
        if filters["sort"] == "name":
            return (name, row["player_id"])
        if filters["sort"] == "age_asc":
            age = row.get("age")
            return (not finite(age), age if finite(age) else 0, name, row["player_id"])
        value = price(row)
        direction = -1 if filters["sort"] == "value_desc" else 1
        return (not finite(value), direction * value if finite(value) else 0, name, row["player_id"])
    return sorted(selected, key=key)
