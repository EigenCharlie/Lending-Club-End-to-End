"""IFRS9 ECL schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ECLRequest(BaseModel):
    """ECL calculation request."""

    pd_multiplier: float = Field(1.0, ge=0.5, le=3.0, description="PD stress multiplier")
    lgd_multiplier: float = Field(1.0, ge=0.5, le=2.0, description="LGD stress multiplier")
    scenario: str = Field(
        "baseline",
        description="Scenario: baseline, mild_stress, adverse, severe (aliases: base->baseline, optimistic->mild_stress)",
    )


class ECLResponse(BaseModel):
    """ECL calculation response."""

    scenario: str
    total_ecl: float
    stage1_ecl: float
    stage2_ecl: float
    stage3_ecl: float
    stage1_count: int
    stage2_count: int
    stage3_count: int
    pd_multiplier: float
    lgd_multiplier: float
