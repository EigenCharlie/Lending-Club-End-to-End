"""Tests for scripts/run_cif_ecl_impact.py."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def cif_workdir(tmp_path, monkeypatch):
    """Set up a tmp working directory with required input parquets."""
    monkeypatch.chdir(tmp_path)

    data_dir = tmp_path / "data" / "processed"
    model_dir = tmp_path / "models"
    data_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)

    # competing_risks_cif.parquet — one ALL row + one grade row
    cif_rows = [
        {
            "grade": "ALL",
            "overall_cif_default_t12m": 0.0300,
            "overall_cif_default_t36m": 0.0800,
            "overall_cif_default_t60m": 0.1239,
            "km_default_prob_t12m": 0.0302,
            "km_default_prob_t36m": 0.0850,
            "km_default_prob_t60m": 0.1690,
            # Per-grade wide columns (only grade A provided)
            "A_cif_default_t12m": 0.0100,
            "A_cif_default_t36m": 0.0350,
            "A_cif_default_t60m": 0.0600,
        },
        {
            "grade": "A",
            "overall_cif_default_t12m": np.nan,
            "overall_cif_default_t36m": np.nan,
            "overall_cif_default_t60m": np.nan,
            "km_default_prob_t12m": np.nan,
            "km_default_prob_t36m": np.nan,
            "km_default_prob_t60m": np.nan,
            "A_cif_default_t12m": 0.0100,
            "A_cif_default_t36m": 0.0350,
            "A_cif_default_t60m": 0.0600,
        },
    ]
    pd.DataFrame(cif_rows).to_parquet(data_dir / "competing_risks_cif.parquet", index=False)

    # ifrs9_ecl_comparison.parquet — two grades
    ecl_rows = [
        {
            "Grade": "A",
            "Avg Loan": 12_000.0,
            "PD_12m": 0.015,
            "PD_lifetime": 0.080,
            "ECL_Stage1": 72.0,
            "ECL_Stage2": 384.0,
        },
        {
            "Grade": "B",
            "Avg Loan": 14_000.0,
            "PD_12m": 0.040,
            "PD_lifetime": 0.180,
            "ECL_Stage1": 224.0,
            "ECL_Stage2": 1_008.0,
        },
    ]
    pd.DataFrame(ecl_rows).to_parquet(data_dir / "ifrs9_ecl_comparison.parquet", index=False)

    # ifrs9_scenario_summary.parquet — optional, baseline row
    sc_rows = [
        {"scenario": "baseline", "total_ecl": 1_000_000.0},
        {"scenario": "severe", "total_ecl": 1_800_000.0},
    ]
    pd.DataFrame(sc_rows).to_parquet(data_dir / "ifrs9_scenario_summary.parquet", index=False)

    # test.parquet — grade + loan_amnt
    test_rows = [
        {"grade": "A", "loan_amnt": 10_000},
        {"grade": "A", "loan_amnt": 12_000},
        {"grade": "B", "loan_amnt": 14_000},
    ]
    pd.DataFrame(test_rows).to_parquet(data_dir / "test.parquet", index=False)

    return tmp_path


def test_cif_ecl_impact_outputs_exist(cif_workdir) -> None:
    from scripts import run_cif_ecl_impact as mod

    rc = mod.main()

    assert rc == 0
    assert (cif_workdir / "data" / "processed" / "cif_ecl_impact.parquet").exists()
    assert (cif_workdir / "models" / "cif_ecl_impact_status.json").exists()


def test_cif_ecl_impact_parquet_schema(cif_workdir) -> None:
    from scripts import run_cif_ecl_impact as mod

    mod.main()

    df = pd.read_parquet(cif_workdir / "data" / "processed" / "cif_ecl_impact.parquet")
    required_cols = {
        "grade",
        "cif_t60m",
        "km_t60m",
        "cf_lifetime",
        "ecl_stage2_km",
        "ecl_stage2_cif",
        "excess_ecl_stage2_per_loan",
        "cif_source",
    }
    assert required_cols.issubset(set(df.columns))
    assert len(df) == 2  # grades A and B
    assert set(df["grade"]) == {"A", "B"}


def test_cif_ecl_impact_correction_direction(cif_workdir) -> None:
    """CIF correction factor must be < 1 (KM overestimates PD)."""
    from scripts import run_cif_ecl_impact as mod

    mod.main()

    df = pd.read_parquet(cif_workdir / "data" / "processed" / "cif_ecl_impact.parquet")
    # CIF < KM → cf_lifetime < 1 → excess > 0
    assert (df["cf_lifetime"] < 1.0).all()
    assert (df["excess_ecl_stage2_per_loan"] > 0).all()


def test_cif_ecl_impact_status_keys(cif_workdir) -> None:
    from scripts import run_cif_ecl_impact as mod

    mod.main()

    status = json.loads(
        (cif_workdir / "models" / "cif_ecl_impact_status.json").read_text(encoding="utf-8")
    )
    assert "schema_version" in status
    assert "overall" in status
    assert "ecl_impact" in status
    overall = status["overall"]
    assert overall["km_t60m"] > overall["cif_t60m"]
    assert 0 < overall["cf_lifetime"] < 1


def test_load_cif_data_returns_overall_and_grade(cif_workdir) -> None:
    from scripts import run_cif_ecl_impact as mod

    cif = mod._load_cif_data()

    assert "ALL" in cif
    assert cif["ALL"]["km_t60m"] == pytest.approx(0.1690)
    assert cif["ALL"]["cif_t60m"] == pytest.approx(0.1239)
    # Grade A was provided in wide columns
    assert "A" in cif
    assert cif["A"]["cif_t60m"] == pytest.approx(0.0600)
