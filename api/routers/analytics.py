"""DuckDB analytics endpoints."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException

from api.config import DATA_DIR, MODEL_DIR
from api.schemas.analytics import QueryRequest, QueryResponse
from api.services.duckdb_service import execute_query, get_table_list

router = APIRouter(prefix="/api/v1", tags=["analytics"])


@router.post("/query", response_model=QueryResponse)
def run_query(req: QueryRequest):
    """Execute a read-only SQL query against DuckDB."""
    try:
        return execute_query(req.sql)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query error: {e}")


@router.get("/tables")
def list_tables():
    """List all available DuckDB tables."""
    return get_table_list()


@router.get("/summary/pipeline")
def pipeline_summary():
    """Get pipeline summary metrics."""
    path = DATA_DIR / "pipeline_summary.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Pipeline summary not found")
    return json.loads(path.read_text())


@router.get("/summary/eda")
def eda_summary():
    """Get EDA summary statistics."""
    path = DATA_DIR / "eda_summary.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="EDA summary not found")
    return json.loads(path.read_text())


@router.get("/summary/conformal")
def conformal_summary():
    """Get conformal prediction policy status."""
    path = MODEL_DIR / "conformal_policy_status.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Conformal policy status not found")
    return json.loads(path.read_text())


@router.get("/summary/model")
def model_summary():
    """Get model comparison metrics."""
    path = DATA_DIR / "model_comparison.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Model comparison not found")
    return json.loads(path.read_text())
