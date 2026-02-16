"""PD prediction endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from api.schemas.loan import LoanInput, PredictionResponse
from api.services.model_service import predict_single

router = APIRouter(prefix="/api/v1", tags=["predictions"])


@router.post("/predict", response_model=PredictionResponse)
def predict(loan: LoanInput):
    """Get PD/LGD/EAD point predictions for a loan."""
    return predict_single(loan.model_dump())
