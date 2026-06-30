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


def resolve_geo_from_places(question: str) -> dict | None:
    """Return {"points": [...], "bbox": {...}} for the first place found in question.

    "bbox" is the authoritative administrative-boundary extent (results.bounds in
    the place JSON), independent of the point list — a single-point place still
    has a real (non-zero-width) administrative bbox.
    """
    q = question.lower()
    for name in _SORTED_NAMES:
        if name in q:
            data = json.loads(_INDEX[name].read_text())
            results = data.get("results", {})
            return {
                "points": results.get("points", []),
                "bbox": results.get("bounds") or {},
            }
    return None


def format_geo_block(points: list[dict], mode: str = "points", bbox: dict | None = None) -> str:
    """Format geo points (or bbox) as a SQL comment block.

    Uses latitude/longitude to match the meteo DB column names. In "bbox" mode,
    prefers the authoritative administrative bbox (rounded to 1 decimal place,
    matching the gold SQL cache's convention) over a point-derived min/max,
    which is structurally narrower for single-point places.
    """
    if mode == "points":
        if not points:
            return ""
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
        if bbox:
            minlat = bbox.get("minlat") or bbox.get("min_lat")
            maxlat = bbox.get("maxlat") or bbox.get("max_lat")
            minlon = bbox.get("minlon") or bbox.get("min_lon")
            maxlon = bbox.get("maxlon") or bbox.get("max_lon")
            if None in (minlat, maxlat, minlon, maxlon):
                bbox = None
        if bbox:
            minlat, maxlat, minlon, maxlon = (
                round(float(v), 1) for v in (minlat, maxlat, minlon, maxlon)
            )
        elif points:
            lats = [float(p["lat"]) for p in points]
            lons = [float(p["lon"]) for p in points]
            minlat, maxlat = round(min(lats), 1), round(max(lats), 1)
            minlon, maxlon = round(min(lons), 1), round(max(lons), 1)
        else:
            return ""
        return (
            f"/* Geographic filter (bbox mode) — coordinates are rounded to 1 decimal place.\n"
            f"   ROUND(CAST(latitude AS DECIMAL), 1) BETWEEN {minlat} AND {maxlat}\n"
            f"   ROUND(CAST(longitude AS DECIMAL), 1) BETWEEN {minlon} AND {maxlon} */\n"
        )
