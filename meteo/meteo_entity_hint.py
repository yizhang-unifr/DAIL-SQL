"""Meteo entity hint: variable name → DB table.column mapping with unit notes."""
from __future__ import annotations

# (keywords, table, column, unit_note) — ordered longest-match first
_VARIABLE_MAP: list[tuple[frozenset[str], str, str, str]] = [
    (
        frozenset({"mean temperature", "average temperature"}),
        "meteo_tmean", "tmean",
        "Kelvin; subtract 273.15 for Celsius",
    ),
    (
        frozenset({"minimum temperature", "min temperature", "coldest", "lowest temperature"}),
        "meteo_tmin", "tmin",
        "Kelvin; subtract 273.15 for Celsius",
    ),
    (
        frozenset({"maximum temperature", "max temperature", "hottest", "warmest",
                   "highest temperature"}),
        "meteo_tmax", "tmax",
        "Kelvin; subtract 273.15 for Celsius",
    ),
    # Generic "temperature" falls through to tmean (most common default)
    (
        frozenset({"temperature"}),
        "meteo_tmean", "tmean",
        "Kelvin; subtract 273.15 for Celsius",
    ),
    (
        frozenset({"wind speed", "wind"}),
        "meteo_windspeedmax", "windspeedmax",
        "m/s",
    ),
    (
        frozenset({"precipitation", "rainfall", "rain"}),
        "meteo_tp", "tp",
        "kg/m²/day",
    ),
    (
        frozenset({"snow depth", "snowfall", "snow"}),
        "meteo_sdmax", "sdmax",
        "metres",
    ),
    (
        frozenset({"elevation", "altitude", "height"}),
        "meteo_elevation", "elevation",
        "metres above sea level",
    ),
]


def build_entity_hint(question: str) -> str:
    """Return a SQL comment block mapping detected variable names to table.column."""
    q = question.lower()
    matched: list[str] = []
    seen_tables: set[str] = set()
    for keywords, table, column, unit in _VARIABLE_MAP:
        if table in seen_tables:
            continue
        if any(kw in q for kw in keywords):
            matched.append(f"   {table}.{column}: {unit}")
            seen_tables.add(table)
    if not matched:
        return ""
    lines = ["/* Variable mapping:"] + matched + ["*/"]
    return "\n".join(lines) + "\n"
