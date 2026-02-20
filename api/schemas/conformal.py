"""Conformal prediction response schemas."""

from __future__ import annotations

from pydantic import Field

from api.schemas.loan import PredictionResponse


class ConformalResponse(PredictionResponse):
    """Prediction with conformal prediction intervals."""

    pd_low: float = Field(..., description="Lower bound of PD interval")
    pd_high: float = Field(..., description="Upper bound of PD interval")
    interval_width: float = Field(..., description="Width of prediction interval")
    confidence_level: float = Field(..., description="Nominal coverage level (e.g. 0.90)")
    expected_loss_conservative: float = Field(..., description="EL using PD_high")
