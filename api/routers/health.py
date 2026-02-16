"""Health check endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from api.config import DUCKDB_PATH, MODEL_DIR, PD_MODEL_FILE

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    """Basic health check."""
    return {"status": "ok"}


@router.get("/ready")
def ready():
    """Readiness check — verifies model and data files exist."""
    checks = {
        "pd_model": (MODEL_DIR / PD_MODEL_FILE).exists(),
        "duckdb": DUCKDB_PATH.exists(),
    }
    all_ready = all(checks.values())
    return {"ready": all_ready, "checks": checks}
