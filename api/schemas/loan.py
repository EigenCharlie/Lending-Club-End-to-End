"""Loan input and prediction response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LoanInput(BaseModel):
    """Input features for a single loan prediction."""

    loan_amnt: float = Field(..., gt=0, description="Loan amount requested")
    annual_inc: float = Field(..., gt=0, description="Annual income")
    dti: float = Field(..., ge=0, description="Debt-to-income ratio")
    int_rate: float = Field(..., ge=0, le=100, description="Interest rate (%)")
    installment: float = Field(0.0, ge=0, description="Monthly installment")
    term: str = Field("36 months", description="Loan term")
    grade: str = Field("C", description="Lending Club grade (A-G)")
    sub_grade: str = Field("C3", description="Sub-grade")
    home_ownership: str = Field("RENT", description="Home ownership status")
    purpose: str = Field("debt_consolidation", description="Loan purpose")
    emp_length: str = Field("5 years", description="Employment length")
    verification_status: str = Field("Not Verified", description="Income verification")
    open_acc: float = Field(10.0, ge=0)
    pub_rec: float = Field(0.0, ge=0)
    revol_bal: float = Field(10000.0, ge=0)
    revol_util: float = Field(50.0, ge=0)
    total_acc: float = Field(20.0, ge=0)
    fico_range_low: float = Field(700.0, ge=300, le=850)
    fico_range_high: float = Field(704.0, ge=300, le=855)
    credit_history_months: float = Field(120.0, ge=0)
    num_delinq_2yrs: float = Field(0.0, ge=0)
    days_since_last_delinq: float | None = Field(None)
    rev_utilization: float = Field(0.5, ge=0)
    loan_to_income: float | None = Field(None, description="Auto-computed if not provided")


class PredictionResponse(BaseModel):
    """PD prediction response."""

    pd_point: float = Field(..., description="Calibrated PD estimate")
    pd_raw: float = Field(..., description="Raw CatBoost PD (uncalibrated)")
    lgd_estimate: float = Field(0.45, description="LGD estimate (Basel standard)")
    ead_estimate: float = Field(..., description="Exposure at default")
    expected_loss: float = Field(..., description="PD x LGD x EAD")
    risk_category: str = Field(..., description="Low/Medium/High/Very High Risk")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""

    loans: list[LoanInput]
