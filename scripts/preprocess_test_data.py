"""Convert test_data.xlsx to pipeline-compatible eval JSON files.

Outputs (in src/DAIL-SQL/data/data_preprocess/):
  test_data_point.json   — rows where sql_variant == 'point'
  test_data_bbox.json    — rows where sql_variant == 'bbox'

Usage (from project root):
    uv run python src/DAIL-SQL/scripts/preprocess_test_data.py
    uv run python src/DAIL-SQL/scripts/preprocess_test_data.py --limit 20
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_XLSX = _PROJECT_ROOT / "data" / "split_config_II_tiered" / "test_data.xlsx"
_DEFAULT_OUT  = Path(__file__).resolve().parents[1] / "data" / "data_preprocess"

GEO_MODE_MAP = {"point": "points", "bbox": "bbox"}


def preprocess(test_xlsx: Path, out_dir: Path, limit: int | None = None) -> dict[str, Path]:
    log.info("Reading %s …", test_xlsx)
    df = pd.read_excel(test_xlsx)
    log.info("  total rows: %d, variants: %s", len(df), df["sql_variant"].value_counts().to_dict())

    df = df[df["augmentation_validation"] == True].copy()
    log.info("  after validation filter: %d rows", len(df))

    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    for variant, geo_mode in GEO_MODE_MAP.items():
        subset = df[df["sql_variant"] == variant].reset_index(drop=True)
        if limit:
            subset = subset.head(limit)

        records = []
        for i, row in subset.iterrows():
            records.append({
                "question_id": int(i),
                "question": str(row["natural_language_question"]).strip(),
                "raw_question": str(row["natural_language_question"]).strip(),
                "db_id": "meteo",
                "evidence": "",
                "SQL": str(row["sql_query"]).strip(),
                "category": str(row.get("category", "")),
                "geo_filter_mode": geo_mode,
                "template_index": int(row.get("template_index", -1)),
            })

        out_path = out_dir / f"test_data_{variant}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        log.info("  wrote %d rows → %s", len(records), out_path)
        written[variant] = out_path

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess test_data.xlsx → eval JSON for DAIL-SQL")
    parser.add_argument("--test_xlsx", default=str(_DEFAULT_XLSX))
    parser.add_argument("--out_dir", default=str(_DEFAULT_OUT))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    written = preprocess(Path(args.test_xlsx), Path(args.out_dir), args.limit)
    print("\nGenerated files:")
    for variant, p in written.items():
        with open(p) as f:
            n = len(json.load(f))
        print(f"  {p}  ({n} rows)")


if __name__ == "__main__":
    main()
