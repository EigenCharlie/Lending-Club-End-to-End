"""Conformal prediction endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from api.schemas.conformal import ConformalResponse
from api.schemas.loan import LoanInput
from api.services.model_service import predict_conformal

router = APIRouter(prefix="/api/v1", tags=["conformal"])
SUPPORTED_ALPHAS = (0.10, 0.05)


def _normalize_alpha(alpha: float) -> float:
    """Accept only levels backed by current conformal artifacts."""
    for supported in SUPPORTED_ALPHAS:
        if abs(alpha - supported) < 1e-9:
            return supported
    allowed = ", ".join(f"{a:.2f}" for a in SUPPORTED_ALPHAS)
    raise HTTPException(
        status_code=422,
        detail=f"alpha no soportado. Valores permitidos: {allowed}",
    )


@router.post("/conformal", response_model=ConformalResponse)
def predict_with_intervals(
    loan: LoanInput,
    alpha: float = Query(
        0.10,
        ge=0.01,
        le=0.50,
        description="Miscoverage rate. Soportado por artefacto actual: 0.10 (90%) o 0.05 (95%).",
    ),
):
    """Get predictions with conformal prediction intervals."""
    alpha_supported = _normalize_alpha(alpha)
    return predict_conformal(loan.model_dump(), alpha=alpha_supported)
