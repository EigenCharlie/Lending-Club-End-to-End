"""Health and readiness response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Liveness response."""

    status: str = Field(..., description="Service liveness status")
    version: str = Field(..., description="API version")


class ReadinessChecks(BaseModel):
    """Required file checks for serving."""

    pd_model: bool = Field(..., description="PD model artifact exists")
    calibrator: bool = Field(..., description="PD calibrator artifact exists")
    duckdb: bool = Field(..., description="DuckDB analytics database exists")


class PreloadChecks(BaseModel):
    """Startup preload status for core serving artifacts."""

    attempted: bool = Field(False, description="Startup preload lifecycle was executed")
    pd_model_loaded: bool = Field(False, description="PD model loaded in startup preload")
    calibrator_loaded: bool = Field(False, description="Calibrator loaded in startup preload")
    error: str | None = Field(None, description="Last preload error summary (if any)")


class ReadinessResponse(BaseModel):
    """Readiness response combining file checks and preload status."""

    ready: bool = Field(..., description="True when all required checks pass")
    checks: ReadinessChecks
    preload: PreloadChecks
