"""Geo adapter: keyword lookup against src/places/ → lat/lon points.

No LLM. No DB. ~1 ms per call.
"""
from __future__ import annotations

import json
from pathlib import Path

_PLACES_DIR = Path(__file__).resolve().parents[3] / "src" / "places"

# Index: normalised place name → json path (sorted longest-first for greedy match)
_INDEX: dict[str, Path] = {}
# regions: bare name + "canton of X" / "region of X" prefixes
for _p in (_PLACES_DIR / "regions").glob("*.json") if (_PLACES_DIR / "regions").exists() else []:
    stem = _p.stem.lower().replace("_", " ")
    _INDEX[stem] = _p
    _INDEX[f"canton of {stem}"] = _p
    _INDEX[f"region of {stem}"] = _p
# cities: "city of X" prefix (longer → wins over bare stem in sorted lookup)
for _p in (_PLACES_DIR / "cities").glob("*.json") if (_PLACES_DIR / "cities").exists() else []:
    stem = _p.stem.lower().replace("_", " ")
    _INDEX[f"city of {stem}"] = _p

_SORTED_NAMES = sorted(_INDEX, key=len, reverse=True)


def resolve_geo_from_places(question: str) -> list[dict] | None:
    """Return [{"lat": ..., "lon": ...}, ...] for the first place found in question."""
    q = question.lower()
    for name in _SORTED_NAMES:
        if name in q:
            data = json.loads(_INDEX[name].read_text())
            return data.get("results", {}).get("points", [])
    return None


def format_geo_block(points: list[dict], mode: str = "points") -> str:
    """Format geo points as a SQL comment block.

    Uses latitude/longitude to match the meteo DB column names.
    """
    if not points:
        return ""
    if mode == "points":
        coords = ", ".join(
            f"({p['lat']}, {p['lon']})" for p in points
        )
        n = len(points)
        return (
            f"/* Geographic filter — {n} coordinate pairs (DO NOT add, remove, or modify any).\n"
            f"   Copy ALL {n} pairs below EXACTLY into your IN clause:\n"
            f"   (ROUND(CAST(latitude AS DECIMAL), 1), ROUND(CAST(longitude AS DECIMAL), 1)) IN ({coords}) */\n"
        )
    else:
        lats = [float(p["lat"]) for p in points]
        lons = [float(p["lon"]) for p in points]
        return (
            f"/* Geographic filter (bbox mode) — coordinates are rounded to 1 decimal place.\n"
            f"   ROUND(CAST(latitude AS DECIMAL), 1) BETWEEN {min(lats)} AND {max(lats)}\n"
            f"   ROUND(CAST(longitude AS DECIMAL), 1) BETWEEN {min(lons)} AND {max(lons)} */\n"
        )
