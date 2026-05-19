"""Build natural-language usage contracts from PostgreSQL column metadata."""

from __future__ import annotations

import os

from runner.execution import _get_pg_connection

_METEO_ALLOWED_TABLES: set[str] = {"landcover_type", "landcover_upscaled"}

_INT2D_CONTRACTS: list[str] = [
    "This column is an integer[][] (2-D integer array). NEVER use UNNEST() — it flattens to individual scalars, not (code, count) pairs.",
    "To iterate all entries use: CROSS JOIN LATERAL GENERATE_SUBSCRIPTS(<alias>.{col}, 1) AS i",
    "Access element code with <alias>.{col}[i][1] and count/pixels with <alias>.{col}[i][2].",
    "For the single dominant (top-ranked) entry only, direct subscript <alias>.{col}[1][1] is correct.",
]


def _is_table_allowed(table_name: str) -> bool:
    if table_name.startswith("meteo_"):
        return True
    return table_name in _METEO_ALLOWED_TABLES


def build_column_contracts(conn=None, schema: str | None = None) -> dict[str, list[str]]:
    if schema is None:
        schema = os.environ.get("DB_SCHEMA", "public")

    own_conn = conn is None
    if own_conn:
        conn = _get_pg_connection()

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    c.relname  AS table_name,
                    a.attname  AS column_name,
                    format_type(a.atttypid, a.atttypmod) AS full_type
                FROM pg_attribute a
                JOIN pg_class     c ON c.oid  = a.attrelid
                JOIN pg_namespace n ON n.oid  = c.relnamespace
                WHERE n.nspname   = %s
                  AND a.attnum    > 0
                  AND NOT a.attisdropped
                ORDER BY c.relname, a.attnum;
                """,
                (schema,),
            )
            rows = cur.fetchall()
    finally:
        if own_conn:
            conn.close()

    contracts: dict[str, list[str]] = {}
    for table_name, col_name, full_type in rows:
        if not _is_table_allowed(table_name):
            continue
        if full_type == "integer[][]":
            contracts[f"{table_name}.{col_name}"] = [
                rule.replace("{col}", col_name) for rule in _INT2D_CONTRACTS
            ]
    return contracts


def format_contracts_for_prompt(contracts: dict[str, list[str]]) -> str:
    if not contracts:
        return "(none)"
    lines: list[str] = []
    for col_key, rules in contracts.items():
        lines.append(f"Column `{col_key}`:")
        for rule in rules:
            lines.append(f"  - {rule}")
    return "\n".join(lines)
