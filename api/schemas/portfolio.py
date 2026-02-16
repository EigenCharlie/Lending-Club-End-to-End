"""Portfolio optimization schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PortfolioRequest(BaseModel):
    """Portfolio optimization request."""

    budget: float = Field(1_000_000, gt=0, description="Total budget to allocate")
    risk_tolerance: float = Field(0.15, ge=0, le=1, description="Maximum portfolio PD")
    max_concentration: float = Field(0.05, ge=0, le=1, description="Max single-loan share")
    use_robust: bool = Field(True, description="Use conformal-robust optimization")
    n_loans: int = Field(100, ge=10, le=10000, description="Number of loans to consider")


class PortfolioResponse(BaseModel):
    """Portfolio optimization results."""

    objective_value: float
    n_funded: int
    total_allocated: float
    portfolio_pd: float
    portfolio_expected_return: float
    portfolio_expected_loss: float
    is_robust: bool
