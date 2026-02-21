"""IFRS9 ECL calculation endpoints."""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException

from api.config import DATA_DIR
from api.schemas.ecl import ECLRequest, ECLResponse

router = APIRouter(prefix="/api/v1", tags=["ifrs9"])


SCENARIO_MULTIPLIERS = {
    "baseline": {"pd_mult": 1.00, "lgd_mult": 1.00},
    "mild_stress": {"pd_mult": 1.15, "lgd_mult": 1.05},
    "adverse": {"pd_mult": 1.35, "lgd_mult": 1.15},
    "severe": {"pd_mult": 1.60, "lgd_mult": 1.30},
}

SCENARIO_ALIASES = {
    "baseline": "baseline",
    "base": "baseline",
    "mild_stress": "mild_stress",
    "optimistic": "mild_stress",
    "adverse": "adverse",
    "severe": "severe",
}


def _normalize_scenario(scenario: str) -> str:
    raw = str(scenario).strip().lower()
    normalized = SCENARIO_ALIASES.get(raw)
    if normalized is not None:
        return normalized
    allowed = ", ".join(sorted(set(SCENARIO_ALIASES.keys())))
    raise HTTPException(
        status_code=422,
        detail=f"scenario no soportado: '{scenario}'. Valores permitidos: {allowed}",
    )


@router.post("/ecl", response_model=ECLResponse)
def calculate_ecl(req: ECLRequest):
    """Calculate IFRS9 Expected Credit Loss under a macroeconomic scenario.

    Computes ECL = PD x LGD x EAD by stage (1/2/3) with scenario-specific
    stress multipliers. Scenarios: baseline, mild_stress, adverse, severe.
    Returns stage-level ECL breakdown and loan counts.
    """
    scenario = _normalize_scenario(req.scenario)

    # Load pre-computed scenario data if available
    scenario_path = DATA_DIR / "ifrs9_scenario_summary.parquet"
    if scenario_path.exists():
        df = pd.read_parquet(scenario_path)
        match = df[df["scenario"] == scenario]
        if len(match) > 0:
            row = match.iloc[0]
            return ECLResponse(
                scenario=scenario,
                total_ecl=float(row.get("total_ecl", 0)),
                stage1_ecl=float(row.get("stage1_ecl", 0)),
                stage2_ecl=float(row.get("stage2_ecl", 0)),
                stage3_ecl=float(row.get("stage3_ecl", 0)),
                stage1_count=int(row.get("stage1_n", 0)),
                stage2_count=int(row.get("stage2_n", 0)),
                stage3_count=int(row.get("stage3_n", 0)),
                pd_multiplier=req.pd_multiplier,
                lgd_multiplier=req.lgd_multiplier,
            )

    # Fallback: compute from pre-computed ECL comparison
    mults = SCENARIO_MULTIPLIERS[scenario]
    ecl_path = DATA_DIR / "ifrs9_ecl_comparison.parquet"
    df = pd.read_parquet(ecl_path)

    stage1_ecl = float(df["ECL_Stage1"].sum()) * mults["pd_mult"] * mults["lgd_mult"]
    stage2_ecl = float(df["ECL_Stage2"].sum()) * mults["pd_mult"] * mults["lgd_mult"]

    return ECLResponse(
        scenario=scenario,
        total_ecl=stage1_ecl + stage2_ecl,
        stage1_ecl=stage1_ecl,
        stage2_ecl=stage2_ecl,
        stage3_ecl=0.0,
        stage1_count=len(df),
        stage2_count=len(df),
        stage3_count=0,
        pd_multiplier=req.pd_multiplier,
        lgd_multiplier=req.lgd_multiplier,
    )
