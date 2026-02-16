"""FastAPI application for credit risk predictions.

Endpoints:
  /health             — Health check
  /ready              — Readiness check
  /api/v1/predict     — PD/LGD/EAD point predictions
  /api/v1/conformal   — Predictions with conformal intervals
  /api/v1/ecl         — IFRS9 ECL calculation
  /api/v1/query       — DuckDB analytics queries
  /api/v1/tables      — List DuckDB tables
  /api/v1/summary/*   — Pre-computed summaries

Run: uv run uvicorn api.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.config import API_TITLE, API_VERSION, CORS_ORIGINS
from api.routers import analytics, conformal, ecl, health, predict


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load models on startup."""
    logger.info("Loading models...")
    try:
        from api.dependencies import get_calibrator, get_pd_model

        get_pd_model()
        get_calibrator()
        logger.success("Models loaded successfully")
    except Exception as e:
        logger.warning(f"Model pre-loading failed: {e}. Will load on first request.")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title=API_TITLE,
    description="ML + Conformal Prediction + Portfolio Optimization for Credit Risk",
    version=API_VERSION,
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(conformal.router)
app.include_router(ecl.router)
app.include_router(analytics.router)
