"""Meteo-domain prompt representation for DAIL-SQL.

Extends SQLPrompt to inject geo, OGF, entity hint, and semantic hint blocks
between schema and question. Activated via --prompt_repr METEO.

meteo_mode controls which blocks are injected:
    "geo-only"  → geo block only
    "ogf"       → geo + OGF block
    "hints"     → geo + OGF + entity hint + semantic hint
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from prompt.PromptReprTemplate import SQLPrompt
from meteo.geo_adapter import format_geo_block

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))


def _format_ogf_block(ogf_json: str) -> str:
    if not ogf_json:
        return ""
    try:
        ogf = json.loads(ogf_json)
    except Exception:
        return ""
    lines = ["/* Ontology-grounded function context:"]
    if tv := ogf.get("target_variable"):
        lines.append(f"   target_variable: {tv}")
    for spec in ogf.get("function_specs", []):
        if sem := spec.get("semantics", ""):
            lines.append(f"   semantics: {sem}")
        if tmpl := spec.get("pseudo_code_template", ""):
            lines.append(f"   formula: {tmpl}")
    lines.append("*/")
    return "\n".join(lines) + "\n"


class MeteoSQLPrompt(SQLPrompt):
    """SQLPrompt extended with per-question geo, OGF, and hint injection.

    Reads meteo_context.json at __init__ time via context_path argument.
    """

    def __init__(self, context_path: str, meteo_mode: str = "ogf", *args, **kwargs):
        super().__init__(*args, **kwargs)
        data = json.loads(Path(context_path).read_text())
        self._ctx: dict = data.get("by_question", {})
        self._meteo_mode = meteo_mode

    def format_question(self, example: dict) -> str:
        from utils.utils import get_sql_for_database
        sqls = get_sql_for_database(example["path_db"])
        prompt_info = self.template_info.format("\n\n".join(sqls))
        prompt_question = self.template_question.format(example["question"])

        key = example["question"].strip().lower()
        ctx = self._ctx.get(key, {})

        extra_blocks: list[str] = []

        geo_pts = ctx.get("geo_points", [])
        if geo_pts:
            extra_blocks.append(format_geo_block(geo_pts, mode="points").rstrip())

        if self._meteo_mode in ("ogf", "hints"):
            ogf_block = _format_ogf_block(ctx.get("ogf_json", ""))
            if ogf_block:
                extra_blocks.append(ogf_block.rstrip())

        if self._meteo_mode == "hints":
            if entity_hint := ctx.get("entity_hint", ""):
                extra_blocks.append(entity_hint.rstrip())
            if semantic_hint := ctx.get("semantic_hint", ""):
                extra_blocks.append(semantic_hint.rstrip())

        return "\n\n".join([prompt_info] + extra_blocks + [prompt_question])
