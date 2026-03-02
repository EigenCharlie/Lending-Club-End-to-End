"""Health check endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response, status

from api.config import API_VERSION, CALIBRATOR_FILE, DUCKDB_PATH, MODEL_DIR, PD_MODEL_FILE
from api.schemas.health import HealthResponse, PreloadChecks, ReadinessChecks, ReadinessResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Basic health check."""
    return HealthResponse(status="ok", version=API_VERSION)


@router.get("/ready", response_model=ReadinessResponse)
def ready(request: Request, response: Response) -> ReadinessResponse:
    """Readiness check — verifies artifacts exist and startup preload succeeded."""
    checks = ReadinessChecks(
        pd_model=(MODEL_DIR / PD_MODEL_FILE).exists(),
        calibrator=(MODEL_DIR / CALIBRATOR_FILE).exists(),
        duckdb=DUCKDB_PATH.exists(),
    )
    app_state = getattr(request.app, "state", None)
    preload_raw = getattr(app_state, "preload_status", {}) or {}
    preload = PreloadChecks(
        attempted=bool(preload_raw.get("attempted", False)),
        pd_model_loaded=bool(preload_raw.get("pd_model_loaded", False)),
        calibrator_loaded=bool(preload_raw.get("calibrator_loaded", False)),
        error=preload_raw.get("error"),
    )
    all_ready = (
        all(checks.model_dump().values())
        and preload.attempted
        and preload.pd_model_loaded
        and preload.calibrator_loaded
    )
    if not all_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(ready=all_ready, checks=checks, preload=preload)
