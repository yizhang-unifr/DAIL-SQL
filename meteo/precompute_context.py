"""Precompute geo + OGF + hints context for all test questions.

Usage (from project root):
    uv run python src/DAIL-SQL/meteo/precompute_context.py \\
        --questions src/DAIL-SQL/data/data_preprocess/test_data_point.json \\
        --output data/meteo_context.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_DAIL_ROOT    = Path(__file__).resolve().parents[1]   # src/DAIL-SQL/
_PROJECT_ROOT = Path(__file__).resolve().parents[3]   # ontology_retriever/

for _p in (str(_DAIL_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from meteo.geo_adapter import resolve_geo_from_places
from meteo.meteo_entity_hint import build_entity_hint
from meteo.meteo_semantic_hint import build_semantic_hint

_COLUMN_CONTRACTS = {
    "meteo_tmean.tmean":               ["numeric", "unit:Kelvin — subtract 273.15 for Celsius"],
    "meteo_tmin.tmin":                 ["numeric", "unit:Kelvin — subtract 273.15 for Celsius"],
    "meteo_tmax.tmax":                 ["numeric", "unit:Kelvin — subtract 273.15 for Celsius"],
    "meteo_windspeedmax.windspeedmax": ["numeric", "unit:m/s"],
    "meteo_tp.tp":                     ["numeric", "unit:kg/m²/day"],
    "meteo_sdmax.sdmax":               ["numeric", "unit:metres"],
    "meteo_elevation.elevation":       ["numeric", "unit:metres above sea level"],
}


def build_context_for_question(question: str, llm_adapter=None) -> dict:
    points = resolve_geo_from_places(question) or []

    ogf_json = ""
    try:
        from text2sql.tools.kg_context_tool import build_kg_context
        ogf_json = build_kg_context(question).ontology_grounded_function_json or ""
    except Exception as e:
        print(f"  OGF failed for {question!r}: {e}", file=sys.stderr)

    return {
        "geo_points":       points,
        "ogf_json":         ogf_json,
        "entity_hint":      build_entity_hint(question),
        "semantic_hint":    build_semantic_hint(question, llm_adapter=llm_adapter),
        "column_contracts": _COLUMN_CONTRACTS,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions",
                    default=str(_DAIL_ROOT / "data" / "data_preprocess" / "test_data_point.json"),
                    help="Path to test JSON file")
    ap.add_argument("--output",
                    default=str(_DAIL_ROOT / "data" / "meteo_context.json"),
                    help="Output meteo_context.json path")
    ap.add_argument("--llm-config", default=str(_PROJECT_ROOT / "config" / "models.yaml"),
                    help="Path to LLM config YAML (used for landcover classification)")
    args = ap.parse_args()

    # Load LLM adapter for landcover semantic hint classification
    llm_adapter = None
    try:
        from meteo.llm_adapter import get_adapter
        llm_adapter = get_adapter()
        print("LLM adapter loaded for landcover classification")
    except Exception as e:
        print(f"[warn] LLM adapter unavailable — landcover hints will use generic fallback: {e}",
              file=sys.stderr)

    items = json.loads(Path(args.questions).read_text())
    by_question: dict = {}
    for item in items:
        q = item["question"].strip()
        key = q.lower()
        if key in by_question:
            continue
        print(f"Processing: {q[:80]}")
        by_question[key] = build_context_for_question(q, llm_adapter=llm_adapter)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps({"by_question": by_question}, indent=2, ensure_ascii=False)
    )
    print(f"\nWritten {len(by_question)} entries → {args.output}")


if __name__ == "__main__":
    main()
