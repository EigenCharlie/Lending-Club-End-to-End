"""DuckDB query execution service with read-only safety."""

from __future__ import annotations

import re

from loguru import logger

from api.dependencies import get_duckdb_conn

# SQL statements that are NOT allowed
FORBIDDEN_PATTERNS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|GRANT|REVOKE|EXEC)\b",
    re.IGNORECASE,
)


def execute_query(sql: str) -> dict:
    """Execute a read-only SQL query against DuckDB.

    Returns dict with columns, data, and row_count.
    """
    # Safety check
    if FORBIDDEN_PATTERNS.search(sql):
        raise ValueError("Only read-only queries (SELECT) are allowed")

    conn = get_duckdb_conn()
    try:
        result = conn.execute(sql).fetchdf()
        columns = result.columns.tolist()
        # Convert to JSON-safe types
        data = result.head(1000).to_dict(orient="records")
        for row in data:
            for k, v in row.items():
                if hasattr(v, "item"):
                    row[k] = v.item()
                elif hasattr(v, "isoformat"):
                    row[k] = v.isoformat()

        logger.info(f"Query returned {len(result)} rows, {len(columns)} columns")
        return {"columns": columns, "data": data, "row_count": len(result)}
    finally:
        conn.close()


def get_table_list() -> list[dict]:
    """List all available tables in DuckDB."""
    conn = get_duckdb_conn()
    try:
        tables = conn.execute(
            "SELECT table_schema, table_name, table_type "
            "FROM information_schema.tables "
            "ORDER BY table_schema, table_name"
        ).fetchdf()
        return tables.to_dict(orient="records")
    finally:
        conn.close()
