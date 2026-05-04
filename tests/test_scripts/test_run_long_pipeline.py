"""Tests for pipeline-first orchestrator behavior."""

from __future__ import annotations

import argparse
import json

from scripts import run_long_pipeline as lp


def _args(**overrides):
    base = {
        "run_tag": "test-run",
        "pipeline_family": "core_canonical",
        "pipeline_profile": None,
        "resume": False,
        "refresh_baseline_on_resume": True,
        "env_file": None,
        "no_rapids": True,
        "no_notebooks": True,
        "stop_on_optional_failure": False,
        "stall_window_minutes": 15,
        "from_step": None,
        "until_step": None,
        "sampling_profile": "champion64safe",
        "comparison_baseline": None,
        "comparison_baseline_run_tag": None,
        "upstream_canonical_run_tag": None,
        "writes_canonical_artifacts": None,
        "pd_config_override": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_build_steps_diagnostics_governance_runs_governance_before_mrm() -> None:
    contract = lp._derive_pipeline_contract(
        pipeline_family="core_canonical",
        pipeline_profile_arg=None,
        sampling_profile="champion64safe",
        writes_canonical_artifacts_arg=None,
        upstream_canonical_run_tag="baseline-run",
    )
    steps = lp.build_steps(
        "run-x",
        include_rapids=False,
        include_notebooks=False,
        pipeline_family="core_canonical",
        pipeline_contract=contract,
        profile_cfg={"execution": {"include_conformal_sensitivity": False}},
    )
    diagnostics_cmd = next(
        cmd for name, _required, cmd in steps if name == "diagnostics_governance"
    )

    assert "generate_pipeline_registries.py" in diagnostics_cmd
    assert "analyze_pd_rare_event_calibration.py --run-tag run-x" in diagnostics_cmd
    assert "generate_governance_status.py" in diagnostics_cmd
    assert "run_monotonicity_audit.py" in diagnostics_cmd
    assert "run_pd_backtesting_suite.py" in diagnostics_cmd
    assert "run_bootstrap_validation_diagnostics.py --run-tag run-x" in diagnostics_cmd
    assert "run_pd_validation_interpretation.py --run-tag run-x" in diagnostics_cmd
    assert "run_calibration_mapping_diagnostics.py --run-tag run-x" in diagnostics_cmd
    assert "run_encoding_stability_audit.py" in diagnostics_cmd
    assert "generate_mrm_report.py --run-tag run-x" in diagnostics_cmd
    assert diagnostics_cmd.index("run_fairness_audit.py") < diagnostics_cmd.index(
        "run_monotonicity_audit.py"
    )
    assert diagnostics_cmd.index("generate_governance_status.py") < diagnostics_cmd.index(
        "generate_mrm_report.py"
    )
    assert "export_streamlit_artifacts.py" not in diagnostics_cmd
    assert "build_pipeline_results.py" not in diagnostics_cmd


def test_build_steps_search_pd_excludes_unrelated_lanes() -> None:
    contract = lp._derive_pipeline_contract(
        pipeline_family="search_pd",
        pipeline_profile_arg=None,
        sampling_profile="mega64plus",
        writes_canonical_artifacts_arg=None,
        upstream_canonical_run_tag="baseline-run",
    )
    steps = lp.build_steps(
        "run-pd",
        include_rapids=True,
        include_notebooks=True,
        sampling_profile="mega64plus",
        pipeline_family="search_pd",
        pipeline_contract=contract,
        profile_cfg={
            "search_space": {
                "pd": {
                    "config_path": "configs/pd_model.smart.yaml",
                    "monotonic_competitor_config": "configs/monotonic_competitor_blockwise_exhaustive.yaml",
                }
            }
        },
    )
    names = [name for name, _required, _cmd in steps]
    assert names == ["preflight", "core_data_pd", "diagnostics_governance"]
    by_name = {name: cmd for name, _required, cmd in steps}
    assert "train_pd_model.py --config configs/pd_model.smart.yaml" in by_name["core_data_pd"]
    assert (
        "search_monotonic_competitor.py --config configs/monotonic_competitor_blockwise_exhaustive.yaml --run-tag run-pd"
        in by_name["core_data_pd"]
    )
    assert "generate_conformal_intervals.py" not in by_name["core_data_pd"]


def test_build_steps_pd_config_override_takes_precedence(tmp_path, monkeypatch) -> None:
    repo = tmp_path
    (repo / "configs").mkdir(parents=True, exist_ok=True)
    (repo / "configs" / "pd_model.smart.yaml").write_text("model: {}", encoding="utf-8")
    (repo / "models" / "search_pd" / "override-run").mkdir(parents=True, exist_ok=True)
    override_rel = "models/search_pd/override-run/pd_model_hpo_local.yaml"
    (repo / override_rel).write_text("model: {}", encoding="utf-8")
    monkeypatch.setattr(lp, "REPO_ROOT", repo)
    contract = lp._derive_pipeline_contract(
        pipeline_family="search_pd",
        pipeline_profile_arg=None,
        sampling_profile="mega64plus",
        writes_canonical_artifacts_arg=None,
        upstream_canonical_run_tag="baseline-run",
    )
    steps = lp.build_steps(
        "run-pd-override",
        include_rapids=False,
        include_notebooks=False,
        sampling_profile="mega64plus",
        pipeline_family="search_pd",
        pipeline_contract=contract,
        profile_cfg={"search_space": {"pd": {"config_path": "configs/pd_model.smart.yaml"}}},
        pd_config_override=override_rel,
    )
    by_name = {name: cmd for name, _required, cmd in steps}
    assert f"train_pd_model.py --config {override_rel}" in by_name["preflight"]
    assert f"train_pd_model.py --config {override_rel}" in by_name["core_data_pd"]


def test_build_steps_search_conformal_runs_sensitivity_outside_core() -> None:
    contract = lp._derive_pipeline_contract(
        pipeline_family="search_conformal",
        pipeline_profile_arg=None,
        sampling_profile="champion64safe",
        writes_canonical_artifacts_arg=None,
        upstream_canonical_run_tag="baseline-run",
    )
    steps = lp.build_steps(
        "run-conf",
        include_rapids=False,
        include_notebooks=False,
        sampling_profile="champion64safe",
        pipeline_family="search_conformal",
        pipeline_contract=contract,
        profile_cfg={
            "execution": {"include_conformal_sensitivity": True},
            "search_space": {
                "conformal": {
                    "partition_candidates": [
                        "score_decile_mondrian",
                        "grade",
                        "grade_x_scoreband_mondrian",
                    ],
                    "partition_probability_sources": ["calibrated", "raw"],
                    "n_score_bins_candidates": [5, 10, 15],
                    "fallback_modes": ["grade_then_global", "global_only"],
                    "score_scale_families": ["none", "bernoulli_sqrt_clipped_0.02"],
                    "alpha_candidates_95": [0.045, 0.05, 0.055],
                    "calibration_fraction": 0.75,
                    "evaluation_scope": "holdout",
                    "calibration_size_fractions": [0.25, 0.50, 1.0],
                    "shrinkback_enabled": True,
                    "group_coverage_floor_enabled": True,
                    "scaled_scores_options": [True, False],
                }
            },
        },
    )
    by_name = {name: cmd for name, _required, cmd in steps}
    conformal_cmd = by_name["core_conformal"]
    assert "--artifact_namespace run-conf" in conformal_cmd
    assert (
        "--partition_candidates score_decile_mondrian,grade,grade_x_scoreband_mondrian"
        in conformal_cmd
    )
    assert "--partition_probability_sources calibrated,raw" in conformal_cmd
    assert "--n_score_bins_candidates 5,10,15" in conformal_cmd
    assert "--fallback_modes grade_then_global,global_only" in conformal_cmd
    assert "--score_scale_families none,bernoulli_sqrt_clipped_0.02" in conformal_cmd
    assert "--alpha_candidates_95 0.045,0.05,0.055" in conformal_cmd
    assert "--calibration_fraction 0.75" in conformal_cmd
    assert "--evaluation_scope holdout" in conformal_cmd
    assert "--shrinkback_enabled 1" in conformal_cmd
    assert "--group_coverage_floor_enabled 1" in conformal_cmd
    assert "--scaled_scores_options True,False" in conformal_cmd
    assert (
        "--intervals-path data/processed/conformal_gap/run-conf/conformal_intervals_mondrian.parquet"
        in conformal_cmd
    )
    assert "--output-dir data/processed/conformal_gap/run-conf" in conformal_cmd
    assert "--artifact-namespace run-conf" in conformal_cmd
    assert "--sensitivity-config configs/conformal_policy_sensitivity.yaml" in conformal_cmd
    assert "--calibration_size_fractions 0.25,0.5,1.0" in conformal_cmd


def test_build_steps_search_portfolio_does_not_retrain_pd(tmp_path, monkeypatch) -> None:
    repo = tmp_path
    (repo / "configs").mkdir(parents=True, exist_ok=True)
    (repo / "configs" / "optimization.yaml").write_text(
        """
portfolio_selection:
  canonical_execution_mode: search
""".strip(),
        encoding="utf-8",
    )
    (repo / "scripts").mkdir(parents=True, exist_ok=True)
    (repo / "scripts" / "optimize_portfolio.py").write_text(
        "parser.add_argument('--max_candidates')", encoding="utf-8"
    )
    (repo / "scripts" / "optimize_portfolio_tradeoff.py").write_text(
        "parser.add_argument('--grid-profile')", encoding="utf-8"
    )
    monkeypatch.setattr(lp, "REPO_ROOT", repo)
    contract = lp._derive_pipeline_contract(
        pipeline_family="search_portfolio",
        pipeline_profile_arg=None,
        sampling_profile="champion64safe",
        writes_canonical_artifacts_arg=None,
        upstream_canonical_run_tag="baseline-run",
    )

    steps = lp.build_steps(
        "run-portfolio",
        include_rapids=False,
        include_notebooks=False,
        sampling_profile="champion64safe",
        pipeline_family="search_portfolio",
        pipeline_contract=contract,
        profile_cfg={
            "search_space": {
                "portfolio": {"max_candidates": 100},
                "tradeoff": {"max_candidates": 100, "grid_profile": "quick"},
            }
        },
    )
    names = [name for name, _required, _cmd in steps]
    assert names == ["preflight", "core_portfolio", "diagnostics_governance"]
    portfolio_cmd = next(cmd for name, _required, cmd in steps if name == "core_portfolio")
    assert "scripts.optimize_portfolio" in portfolio_cmd
    assert "scripts.optimize_portfolio_tradeoff" in portfolio_cmd
    assert "scripts.select_economic_portfolio_policy" in portfolio_cmd
    assert "train_pd_model.py" not in portfolio_cmd


def test_build_steps_paper2_e2e_explicitly_runs_survival() -> None:
    contract = lp._derive_pipeline_contract(
        pipeline_family="paper2_e2e",
        pipeline_profile_arg=None,
        sampling_profile="champion64safe",
        writes_canonical_artifacts_arg=None,
        upstream_canonical_run_tag="baseline-run",
    )
    steps = lp.build_steps(
        "run-paper2",
        include_rapids=False,
        include_notebooks=False,
        sampling_profile="champion64safe",
        pipeline_family="paper2_e2e",
        pipeline_contract=contract,
        profile_cfg={},
    )
    names = [name for name, _required, _cmd in steps]
    assert names == [
        "preflight",
        "core_ts",
        "paper2_survival",
        "core_ifrs9",
        "diagnostics_governance",
    ]
    by_name = {name: cmd for name, _required, cmd in steps}
    assert "forecast_default_rates.py" in by_name["core_ts"]
    assert "run_survival_analysis.py" in by_name["paper2_survival"]
    assert "train_lgd_ead.py" in by_name["paper2_survival"]
    assert "run_ifrs9_sensitivity.py" in by_name["core_ifrs9"]


def test_build_steps_search_paper2_ifrs9_matches_paper2_scope() -> None:
    contract = lp._derive_pipeline_contract(
        pipeline_family="search_paper2_ifrs9",
        pipeline_profile_arg=None,
        sampling_profile="mega64safe",
        writes_canonical_artifacts_arg=None,
        upstream_canonical_run_tag="baseline-run",
    )
    steps = lp.build_steps(
        "run-paper2-search",
        include_rapids=False,
        include_notebooks=False,
        sampling_profile="mega64safe",
        pipeline_family="search_paper2_ifrs9",
        pipeline_contract=contract,
        profile_cfg={
            "search_space": {
                "survival": {
                    "full_data": True,
                    "rsf_n_estimators": 300,
                    "rsf_max_depth": 16,
                    "rsf_sample_size": 650000,
                    "rsf_n_jobs": 16,
                }
            }
        },
    )
    names = [name for name, _required, _cmd in steps]
    assert names == [
        "preflight",
        "core_ts",
        "paper2_survival",
        "core_ifrs9",
        "diagnostics_governance",
    ]
    by_name = {name: cmd for name, _required, cmd in steps}
    assert "forecast_default_rates.py" in by_name["core_ts"]
    assert (
        "run_survival_analysis.py --full-data --rsf_n_estimators 300" in by_name["paper2_survival"]
    )
    assert "--rsf_max_depth 16" in by_name["paper2_survival"]
    assert "--rsf_sample_size 650000" in by_name["paper2_survival"]
    assert "--rsf_n_jobs 16" in by_name["paper2_survival"]
    assert "run_ifrs9_sensitivity.py" in by_name["core_ifrs9"]
    assert "export_streamlit_artifacts.py" not in by_name["core_ifrs9"]


def test_build_steps_core_canonical_forces_frozen_scope_and_no_bundle(
    tmp_path, monkeypatch
) -> None:
    repo = tmp_path
    (repo / "configs").mkdir(parents=True, exist_ok=True)
    (repo / "configs" / "pd_model.champion.yaml").write_text("{}", encoding="utf-8")
    (repo / "configs" / "optimization.yaml").write_text(
        """
portfolio_selection:
  canonical_execution_mode: freeze_if_available
  frozen_champion_policy_path: models/champion_portfolio_policy.json
""".strip(),
        encoding="utf-8",
    )
    (repo / "models").mkdir(parents=True, exist_ok=True)
    (repo / "models" / "champion_portfolio_policy.json").write_text("{}", encoding="utf-8")
    (repo / "scripts").mkdir(parents=True, exist_ok=True)
    (repo / "scripts" / "optimize_portfolio.py").write_text(
        "parser.add_argument('--max_candidates')", encoding="utf-8"
    )
    (repo / "scripts" / "optimize_portfolio_tradeoff.py").write_text(
        "parser.add_argument('--grid-profile')", encoding="utf-8"
    )
    monkeypatch.setattr(lp, "REPO_ROOT", repo)
    contract = lp._derive_pipeline_contract(
        pipeline_family="core_canonical",
        pipeline_profile_arg=None,
        sampling_profile="champion64safe",
        writes_canonical_artifacts_arg=None,
        upstream_canonical_run_tag="baseline-run",
    )

    steps = lp.build_steps(
        "canonical-run",
        include_rapids=False,
        include_notebooks=False,
        sampling_profile="champion64safe",
        pipeline_family="core_canonical",
        pipeline_contract=contract,
        profile_cfg={},
    )
    by_name = {name: cmd for name, _required, cmd in steps}
    assert "--config configs/pd_model.champion.yaml" in by_name["core_data_pd"]
    assert "scripts.optimize_portfolio_tradeoff" not in by_name["core_portfolio"]
    assert "scripts.select_economic_portfolio_policy" not in by_name["core_portfolio"]
    assert "update_champion_registry.py" in by_name["publication_exports"]
    assert "build_champion_search_bundle.py" not in by_name["publication_exports"]


def test_build_steps_research_notebooks_avoids_redundant_paper_suite_execution() -> None:
    contract = lp._derive_pipeline_contract(
        pipeline_family="research_labs",
        pipeline_profile_arg=None,
        sampling_profile="mega64plus",
        writes_canonical_artifacts_arg=None,
        upstream_canonical_run_tag="baseline-run",
    )
    steps = lp.build_steps(
        "run-notebooks",
        include_rapids=False,
        include_notebooks=True,
        sampling_profile="mega64plus",
        pipeline_family="research_labs",
        pipeline_contract=contract,
        profile_cfg={},
    )
    notebooks_cmd = next(cmd for name, _required, cmd in steps if name == "research_notebooks")
    assert "run_all_notebooks.py" in notebooks_cmd
    assert "extract_notebook_images.py" in notebooks_cmd
    assert "run_paper_notebook_suite.py" not in notebooks_cmd


def test_profile_default_comparison_baseline_run_tag_is_resolved(tmp_path, monkeypatch) -> None:
    comparisons = tmp_path / "reports" / "run_comparisons" / "baseline-tag"
    comparisons.mkdir(parents=True)
    (comparisons / "baseline_snapshot.json").write_text("{}", encoding="utf-8")

    profile_cfg = {"defaults": {"comparison_baseline_run_tag": "baseline-tag"}}
    monkeypatch.setattr(lp, "REPO_ROOT", tmp_path)

    resolved = lp._resolve_comparison_baseline_from_profile(profile_cfg)
    assert resolved == (comparisons / "baseline_snapshot.json").resolve()


def test_preflight_validates_pd_config_before_long_run() -> None:
    contract = lp._derive_pipeline_contract(
        pipeline_family="search_pd",
        pipeline_profile_arg=None,
        sampling_profile="mega64plus",
        writes_canonical_artifacts_arg=None,
        upstream_canonical_run_tag="baseline-run",
    )
    steps = lp.build_steps(
        "run-preflight",
        include_rapids=False,
        include_notebooks=False,
        pipeline_family="search_pd",
        pipeline_contract=contract,
    )
    preflight_cmd = next(cmd for name, _required, cmd in steps if name == "preflight")
    assert "scripts/train_pd_model.py --config" in preflight_cmd
    assert "--validate-only" in preflight_cmd


def test_build_steps_profile_can_route_or_phases_to_rapids(tmp_path, monkeypatch) -> None:
    repo = tmp_path
    (repo / "configs").mkdir(parents=True, exist_ok=True)
    (repo / "configs" / "optimization.yaml").write_text(
        """
portfolio_selection:
  canonical_execution_mode: search
""".strip(),
        encoding="utf-8",
    )
    (repo / "scripts").mkdir(parents=True, exist_ok=True)
    (repo / "scripts" / "optimize_portfolio.py").write_text(
        "parser.add_argument('--max_candidates')", encoding="utf-8"
    )
    (repo / "scripts" / "optimize_portfolio_tradeoff.py").write_text(
        "parser.add_argument('--grid-profile')", encoding="utf-8"
    )
    monkeypatch.setattr(lp, "REPO_ROOT", repo)
    monkeypatch.setattr(lp, "_resolve_rapids_python_cmd", lambda _cfg: "rapids-python")
    contract = lp._derive_pipeline_contract(
        pipeline_family="research_labs",
        pipeline_profile_arg=None,
        sampling_profile="mega64plus",
        writes_canonical_artifacts_arg=None,
        upstream_canonical_run_tag="baseline-run",
    )

    steps = lp.build_steps(
        "labs-max",
        include_rapids=True,
        include_notebooks=True,
        sampling_profile="mega64plus",
        pipeline_family="research_labs",
        pipeline_contract=contract,
        profile_cfg={
            "resource_policy": {
                "lgd_ead_backend": "gpu_if_stable",
                "portfolio_backend": "cuopt",
                "tradeoff_backend": "cuopt",
                "selector_backend": "cuopt",
                "ab_backend": "cuopt",
                "cate_backend": "cuopt",
                "ifrs9_mc_backend": "gpu_research",
            },
            "search_space": {
                "cate_portfolio": {"max_candidates": 180000},
                "rapids": {"ifrs9_mc": {"n_scenarios": 8192, "chunk_size": 256}},
            },
        },
    )

    by_name = {name: cmd for name, _required, cmd in steps}
    assert (
        "rapids-python -u -m scripts.optimize_cate_portfolio" in by_name["research_cate_portfolio"]
    )
    assert (
        "run_ifrs9_monte_carlo_gpu.py --n-scenarios 8192 --chunk-size 256"
        in by_name["research_rapids"]
    )


def test_main_rejects_core_run_without_explicit_baseline(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(lp, "REPO_ROOT", tmp_path)
    missing_registry = tmp_path / "missing_baseline_registry.json"
    monkeypatch.setattr(lp, "BASELINE_REGISTRY_PATH", missing_registry)
    monkeypatch.setattr(lp, "PRIMARY_BASELINE_REGISTRY_PATH", missing_registry)
    monkeypatch.setattr(lp, "LEGACY_BASELINE_REGISTRY_PATH", missing_registry)
    monkeypatch.setattr(lp, "parse_args", lambda: _args(run_tag="2026-03-04-C-core-balanced"))
    try:
        lp.main()
    except ValueError as exc:
        assert "explicit comparison baseline" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing explicit baseline on core run")


def test_main_core_run_uses_registry_baseline_when_cli_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(lp, "REPO_ROOT", tmp_path)
    registry_path = tmp_path / "configs" / "baselines" / "core_official_baseline.json"
    baseline_run_tag = "fixed-baseline"
    baseline_snapshot = (
        tmp_path / "reports" / "run_comparisons" / baseline_run_tag / "baseline_snapshot.json"
    )
    baseline_snapshot.parent.mkdir(parents=True, exist_ok=True)
    baseline_snapshot.write_text("{}", encoding="utf-8")
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps({"official_run_tag": baseline_run_tag}), encoding="utf-8")
    monkeypatch.setattr(lp, "BASELINE_REGISTRY_PATH", registry_path)
    monkeypatch.setattr(lp, "PRIMARY_BASELINE_REGISTRY_PATH", registry_path)
    monkeypatch.setattr(lp, "LEGACY_BASELINE_REGISTRY_PATH", registry_path)
    monkeypatch.setattr(lp, "parse_args", lambda: _args(run_tag="2026-03-04-C-core-balanced"))
    monkeypatch.setattr(lp, "load_completed_ok", lambda *args, **kwargs: False)
    monkeypatch.setattr(lp, "run_step", lambda *args, **kwargs: 0)

    exit_code = lp.main()
    assert exit_code == 0

    run_info = json.loads(
        (
            tmp_path / "reports" / "run_logs" / "2026-03-04-C-core-balanced" / "run_info.json"
        ).read_text(encoding="utf-8")
    )
    assert run_info["comparison_baseline_source"] == "registry_default"
    assert str(baseline_snapshot) == run_info["comparison_baseline_path"]


def test_main_optional_failure_with_stop_flag_sets_nonzero_exit(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(lp, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        lp, "parse_args", lambda: _args(run_tag="test-run-stop", stop_on_optional_failure=True)
    )
    monkeypatch.setattr(lp, "load_completed_ok", lambda *args, **kwargs: False)

    def fake_run_step(
        _run_tag: str,
        step: str,
        _command: str,
        *,
        required: bool,
        step_eta_default_seconds: float | None,
        stall_window_seconds: int,
        resume_subphases: bool,
        pipeline_contract: dict[str, object] | None = None,
    ) -> int:
        _ = step_eta_default_seconds, stall_window_seconds, resume_subphases, pipeline_contract
        if required:
            return 0
        if step == "core_conformal":
            return 2
        return 0

    monkeypatch.setattr(lp, "run_step", fake_run_step)

    exit_code = lp.main()
    summary_path = tmp_path / "reports" / "run_logs" / "test-run-stop" / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert exit_code == 1
    assert summary["failed_required"] is False
    assert summary["stopped_on_optional_failure"] is True
    assert summary["failed_steps"] == ["core_conformal"]


def test_main_runs_selected_step_window_only(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(lp, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        lp,
        "parse_args",
        lambda: _args(
            run_tag="test-run-window",
            pipeline_family="paper2_e2e",
            from_step="paper2_survival",
            until_step="diagnostics_governance",
        ),
    )
    monkeypatch.setattr(lp, "load_completed_ok", lambda *args, **kwargs: False)
    seen_steps: list[str] = []

    def fake_run_step(
        _run_tag: str,
        step: str,
        _command: str,
        *,
        required: bool,
        step_eta_default_seconds: float | None,
        stall_window_seconds: int,
        resume_subphases: bool,
        pipeline_contract: dict[str, object] | None = None,
    ) -> int:
        _ = (
            required,
            step_eta_default_seconds,
            stall_window_seconds,
            resume_subphases,
            pipeline_contract,
        )
        seen_steps.append(step)
        return 0

    monkeypatch.setattr(lp, "run_step", fake_run_step)

    exit_code = lp.main()

    assert exit_code == 0
    assert seen_steps == ["paper2_survival", "core_ifrs9", "diagnostics_governance"]


def test_resume_refreshes_baseline_snapshot_when_preflight_skipped(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(lp, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        lp,
        "parse_args",
        lambda: _args(run_tag="test-run-resume-refresh", resume=True),
    )

    def _completed(_run_tag: str, step: str) -> bool:
        return step == "preflight"

    monkeypatch.setattr(lp, "load_completed_ok", _completed)
    refreshed = {"value": False}

    def _refresh(run_tag: str) -> bool:
        refreshed["value"] = run_tag == "test-run-resume-refresh"
        return True

    monkeypatch.setattr(lp, "refresh_baseline_snapshot", _refresh)
    seen_steps: list[str] = []

    def fake_run_step(
        _run_tag: str,
        step: str,
        _command: str,
        *,
        required: bool,
        step_eta_default_seconds: float | None,
        stall_window_seconds: int,
        resume_subphases: bool,
        pipeline_contract: dict[str, object] | None = None,
    ) -> int:
        _ = (
            required,
            step_eta_default_seconds,
            stall_window_seconds,
            resume_subphases,
            pipeline_contract,
        )
        seen_steps.append(step)
        return 0

    monkeypatch.setattr(lp, "run_step", fake_run_step)
    exit_code = lp.main()

    assert exit_code == 0
    assert refreshed["value"] is True
    assert "preflight" not in seen_steps
