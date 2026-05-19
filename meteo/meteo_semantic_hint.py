"""Meteo semantic hint: classify question access pattern → SQL guidance block."""
from __future__ import annotations

# (name, trigger_keywords, hint_text) — checked in order; first match wins
_PATTERNS: list[tuple[str, frozenset[str], str]] = [
    (
        "landcover_dominant",
        frozenset({"landcover", "land cover", "land-cover"}),
        "Pattern: landcover_dominant — "
        "Step 1: CTE to aggregate the metric per (latitude, longitude) for the time window. "
        "Step 2: CTE to identify the extreme value (MAX or MIN) of that metric. "
        "Step 3: CTE to filter to the grid(s) matching the extreme value. "
        "Step 4: JOIN those grids with landcover_upscaled on ROUND(CAST(latitude AS DECIMAL),1); "
        "CROSS JOIN LATERAL GENERATE_SUBSCRIPTS(ranks,1) AS i; "
        "JOIN landcover_type ON level3_code = ranks[i][1]; "
        "GROUP BY latitude, longitude, level1_code, level1_label; SUM(ranks[i][2]) AS pixel_count. "
        "Step 5: RANK() OVER(PARTITION BY latitude, longitude ORDER BY pixel_count DESC) to find the dominant level1 type per grid. "
        "Step 6: JOIN back to retrieve the metric value and output latitude, longitude, level1_code, level1_label, pixel_count, and the metric value. "
        "ORDER BY latitude, longitude.",
    ),
    (
        "heatwave_event",
        frozenset({"heatwave", "heat wave", "consecutive", "streak", "in a row",
                   "run of days"}),
        "Pattern: heatwave_event — use a CTE with ROW_NUMBER() to identify "
        "consecutive day runs above threshold.",
    ),
    (
        "anomaly_trend",
        frozenset({"anomaly", "deviation", "trend", "change", "increased",
                   "decreased", "compared to average", "compared to the average"}),
        "Pattern: anomaly_trend — use a CTE for the long-term baseline, "
        "then compute delta per year via GROUP BY EXTRACT(YEAR FROM time).",
    ),
    (
        "threshold_count",
        frozenset({"how many days", "number of days", "days above", "days below",
                   "exceeded", "warm days", "hot days", "cold days",
                   "frost days", "heat days"}),
        "Pattern: threshold_count — use COUNT(*) WHERE col > threshold.",
    ),
    (
        "spatial_extreme",
        frozenset({"highest", "lowest", "maximum", "minimum", "where has",
                   "which location", "which city", "which region",
                   "coordinates of", "hottest location", "coldest location"}),
        "Pattern: spatial_extreme — SELECT latitude, longitude WHERE col = "
        "(SELECT MAX/MIN(col) FROM ... WHERE ...); use a correlated subquery.",
    ),
    (
        "seasonal_pattern",
        frozenset({"by month", "monthly", "seasonal", "winter", "summer",
                   "spring", "autumn", "fall", "per season"}),
        "Pattern: seasonal_pattern — GROUP BY EXTRACT(MONTH FROM time); "
        "map months to seasons with CASE if needed.",
    ),
    (
        "spatial_comparison",
        frozenset({"compare", "between regions", "difference between",
                   "versus", " vs "}),
        "Pattern: spatial_comparison — use CTEs or CASE per region; "
        "JOIN on time for multi-region comparison.",
    ),
    (
        "multi_table_join",
        frozenset({"elevation and", "and elevation", "snow depth and",
                   "temperature and wind", "precipitation and temperature"}),
        "Pattern: multi_table_join — JOIN two meteo_* tables on (latitude, longitude) "
        "or (latitude, longitude, time); elevation table has no time column.",
    ),
    (
        "climatological_baseline",
        frozenset({"overall", "entire period", "all years", "historical",
                   "long-term", "climate normal", "climatological"}),
        "Pattern: climatological_baseline — aggregate over the full dataset "
        "without year filter; use AVG over all available years.",
    ),
    (
        "temporal_average",
        frozenset({"average", "mean", "avg"}),
        "Pattern: temporal_average — use AVG() with GROUP BY (latitude, longitude); "
        "filter by EXTRACT(YEAR/MONTH/WEEK FROM time) as required.",
    ),
]


def build_semantic_hint(question: str) -> str:
    """Return a SQL pattern guidance comment for the first matched access pattern."""
    q = question.lower()
    for _name, keywords, hint in _PATTERNS:
        if any(kw in q for kw in keywords):
            return f"/* SQL pattern guidance:\n   {hint}\n*/\n"
    return ""
