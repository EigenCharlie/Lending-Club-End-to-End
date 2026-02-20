"""MCP Server for Credit Risk Data Access.

Provides tools and resources for AI agents (Claude, ChatGPT) to query
the lending club credit risk database and retrieve model metrics.

Run: uv run python -m mcp_server.server
"""

from __future__ import annotations

import json
import re

import duckdb
from mcp.server.fastmcp import FastMCP

from mcp_server.config import DATA_DIR, DUCKDB_PATH, MODEL_DIR

mcp = FastMCP("lending-club-risk")

FORBIDDEN_SQL = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|GRANT|REVOKE|EXEC)\b",
    re.IGNORECASE,
)


@mcp.tool()
def query_data(sql: str) -> str:
    """Execute a read-only SQL query against the credit risk DuckDB database.

    Available schemas:
    - main_staging: Raw parquet views (stg_loan_master, stg_test_predictions, etc.)
    - main_credit_risk: Mart tables (fct_loan_risk_assessment, dim_loan_grades, etc.)
    - main_analytics: Analysis tables (fct_model_performance_by_cohort, fct_conformal_coverage)
    - main_feature_store: Feature store OBT (obt_loan_features)

    Example: SELECT grade, count(*), avg(default_flag) FROM main_staging.stg_loan_master GROUP BY grade
    """
    if FORBIDDEN_SQL.search(sql):
        return "Error: Only read-only SELECT queries are allowed."

    try:
        conn = duckdb.connect(str(DUCKDB_PATH), read_only=True)
        result = conn.execute(sql).fetchdf()
        conn.close()
        # Limit output size
        if len(result) > 100:
            result = result.head(100)
            return f"Showing first 100 of {len(result)} rows:\n\n{result.to_markdown(index=False)}"
        return result.to_markdown(index=False)
    except Exception as e:
        return f"Query error: {e}"


@mcp.tool()
def get_model_metrics(component: str) -> str:
    """Get key performance metrics for a pipeline component.

    Args:
        component: One of: pd_model, conformal, survival, causal, ifrs9, portfolio, eda
    """
    try:
        if component == "pd_model":
            data = json.loads((DATA_DIR / "model_comparison.json").read_text())
            return json.dumps(data, indent=2)
        elif component == "conformal":
            data = json.loads((MODEL_DIR / "conformal_policy_status.json").read_text())
            return json.dumps(data, indent=2)
        elif component == "eda":
            data = json.loads((DATA_DIR / "eda_summary.json").read_text())
            return json.dumps(data, indent=2)
        elif component == "pipeline":
            data = json.loads((DATA_DIR / "pipeline_summary.json").read_text())
            return json.dumps(data, indent=2)
        elif component in ("survival", "causal", "ifrs9", "portfolio"):
            data = json.loads((DATA_DIR / "pipeline_summary.json").read_text())
            if component in data:
                return json.dumps(data[component], indent=2)
            return f"Metrics for '{component}' available in pipeline summary."
        else:
            return f"Unknown component '{component}'. Use: pd_model, conformal, survival, causal, ifrs9, portfolio, eda, pipeline"
    except FileNotFoundError:
        return f"Metrics file not found for '{component}'."


@mcp.tool()
def list_datasets() -> str:
    """List all available parquet datasets with their shapes and column names."""
    import os

    datasets = []
    for f in sorted(DATA_DIR.glob("*.parquet")):
        size_mb = os.path.getsize(f) / (1024 * 1024)
        try:
            import pandas as pd

            df = pd.read_parquet(f)
            datasets.append(
                f"- **{f.name}**: {df.shape[0]} rows x {df.shape[1]} cols ({size_mb:.1f} MB)"
            )
        except Exception:
            datasets.append(f"- **{f.name}**: ({size_mb:.1f} MB)")
    return "\n".join(datasets)


@mcp.tool()
def list_tables() -> str:
    """List all tables and views in the DuckDB database."""
    try:
        conn = duckdb.connect(str(DUCKDB_PATH), read_only=True)
        tables = conn.execute(
            "SELECT table_schema, table_name, table_type "
            "FROM information_schema.tables ORDER BY 1, 2"
        ).fetchdf()
        conn.close()
        return tables.to_markdown(index=False)
    except Exception as e:
        return f"Error: {e}"


@mcp.resource("risk://model-contract")
def model_contract() -> str:
    """PD model feature contract — defines expected inputs, types, and ranges."""
    path = MODEL_DIR / "pd_model_contract.json"
    if path.exists():
        return path.read_text()
    return '{"error": "Model contract not found"}'


@mcp.resource("risk://project-summary")
def project_summary() -> str:
    """Executive summary of the credit risk pipeline and key metrics."""
    path = DATA_DIR / "pipeline_summary.json"
    if path.exists():
        return path.read_text()
    return '{"error": "Pipeline summary not found"}'


@mcp.resource("risk://conformal-policy")
def conformal_policy() -> str:
    """Conformal prediction policy status — coverage checks and alerts."""
    path = MODEL_DIR / "conformal_policy_status.json"
    if path.exists():
        return path.read_text()
    return '{"error": "Conformal policy not found"}'


if __name__ == "__main__":
    mcp.run()
