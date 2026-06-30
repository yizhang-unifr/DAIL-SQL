"""Run multiple DAIL-SQL ablation modes sequentially.

Examples:
  uv run src/DAIL-SQL/run/run_ablation.py --modes geo,hints,full --fewshot --fewshot-k 1 --end -1 --export-xlsx
  uv run src/DAIL-SQL/run/run_ablation.py --modes baseline,geo,ogf,hints,full --fewshot --fewshot-k 3 --end -1
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]  # ontology_retriever/
_VALID_MODES = ("baseline", "geo", "ogf", "hints", "full")


def parse_args():
    p = argparse.ArgumentParser(description="Run DAIL-SQL ablation suite sequentially.")
    p.add_argument("--modes", default="geo,hints,full",
                   help="Comma-separated ablation modes (default: geo,hints,full)")
    p.add_argument("--fewshot", action="store_true", default=False)
    p.add_argument("--fewshot-k", type=int, default=3)
    p.add_argument("--end", type=int, default=10)
    p.add_argument("--dataset", default="")
    p.add_argument("--geo-anchor", default="points", choices=["points", "bbox"])
    p.add_argument("--export-xlsx", action="store_true")
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--meteo-context", default="")
    p.add_argument("--llm-config", default="")
    return p.parse_args()


def main():
    args = parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    invalid = [m for m in modes if m not in _VALID_MODES]
    if invalid:
        sys.exit(f"Unknown mode(s): {invalid}. Valid: {list(_VALID_MODES)}")

    print("=" * 60)
    print(f"DAIL-SQL ablation suite: {' → '.join(modes)}")
    print(f"fewshot={args.fewshot}  k={args.fewshot_k}  end={args.end}")
    print("=" * 60)

    base_cmd = [
        "uv", "run", "src/DAIL-SQL/run/run_meteo.py",
        "--end", str(args.end),
        "--n", str(args.n),
        "--temperature", str(args.temperature),
        "--geo-anchor", args.geo_anchor,
    ]
    if args.fewshot:
        base_cmd += ["--fewshot", "--fewshot-k", str(args.fewshot_k)]
    if args.export_xlsx:
        base_cmd.append("--export-xlsx")
    if args.dataset:
        base_cmd += ["--dataset", args.dataset]
    if args.meteo_context:
        base_cmd += ["--meteo-context", args.meteo_context]
    if args.llm_config:
        base_cmd += ["--llm-config", args.llm_config]

    for mode in modes:
        print(f"\n{'─' * 60}")
        print(f"Mode: {mode}")
        print("─" * 60)
        rc = subprocess.run(base_cmd + ["--ablation", mode], cwd=_ROOT).returncode
        if rc != 0:
            sys.exit(f"Mode '{mode}' failed with exit code {rc}")

    print("\n" + "=" * 60)
    print("All modes complete.")


if __name__ == "__main__":
    main()
