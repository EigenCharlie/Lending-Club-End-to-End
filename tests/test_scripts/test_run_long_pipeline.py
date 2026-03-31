"""Tests for long-run orchestrator behavior."""

from __future__ import annotations

import argparse
import json

from scripts import run_long_pipeline as lp


def test_build_steps_post_core_runs_governance_before_mrm() -> None:
    steps = lp.build_steps("run-x", include_rapids=False, include_notebooks=False)
    post_core_cmd = next(cmd for name, _required, cmd in steps if name == "post_core")

    assert "validate_conformal_policy.py --run-tag run-x" in post_core_cmd
    assert post_core_cmd.index("validate_conformal_policy.py") < post_core_cmd.index(
        "run_ifrs9_sensitivity.py"
    )
    assert "run_ifrs9_diagnostics.py --run-tag run-x" in post_core_cmd
    assert "analyze_pd_rare_event_calibration.py --run-tag run-x" in post_core_cmd
    assert "generate_governance_status.py" in post_core_cmd
    assert "run_monotonicity_audit.py" in post_core_cmd
    assert "run_pd_backtesting_suite.py" in post_core_cmd
    assert "run_bootstrap_validation_diagnostics.py --run-tag run-x" in post_core_cmd
    assert "run_pd_validation_interpretation.py --run-tag run-x" in post_core_cmd
    assert "run_calibration_mapping_diagnostics.py --run-tag run-x" in post_core_cmd
    assert "run_encoding_stability_audit.py" in post_core_cmd
    assert "--run-tag run-x" in post_core_cmd
    assert "generate_paper_grade_protocol.py" in post_core_cmd
    assert "generate_mrm_report.py" in post_core_cmd
    assert "generate_mrm_report.py --run-tag run-x" in post_core_cmd
    assert post_core_cmd.index("run_ifrs9_sensitivity.py") < post_core_cmd.index(
        "run_ifrs9_diagnostics.py"
    )
    assert post_core_cmd.index("build_pipeline_results.py") < post_core_cmd.index(
        "analyze_pd_rare_event_calibration.py"
    )
    assert post_core_cmd.index("run_fairness_audit.py") < post_core_cmd.index(
        "run_monotonicity_audit.py"
    )
    assert post_core_cmd.index("run_monotonicity_audit.py") < post_core_cmd.index(
        "run_pd_backtesting_suite.py"
    )
    assert post_core_cmd.index("run_pd_backtesting_suite.py") < post_core_cmd.index(
        "run_bootstrap_validation_diagnostics.py"
    )
    assert post_core_cmd.index("run_bootstrap_validation_diagnostics.py") < post_core_cmd.index(
        "run_pd_validation_interpretation.py"
    )
    assert post_core_cmd.index("run_pd_validation_interpretation.py") < post_core_cmd.index(
        "run_calibration_mapping_diagnostics.py"
    )
    assert post_core_cmd.index("run_calibration_mapping_diagnostics.py") < post_core_cmd.index(
        "run_encoding_stability_audit.py"
    )
    assert post_core_cmd.index("run_pd_backtesting_suite.py") < post_core_cmd.index(
        "run_encoding_stability_audit.py"
    )
    assert post_core_cmd.index("run_pd_backtesting_suite.py") < post_core_cmd.index(
        "generate_governance_status.py"
    )
    assert post_core_cmd.index("run_encoding_stability_audit.py") < post_core_cmd.index(
        "generate_governance_status.py"
    )
    assert post_core_cmd.index("generate_governance_status.py") < post_core_cmd.index(
        "generate_paper_grade_protocol.py"
    )
    assert post_core_cmd.index("generate_paper_grade_protocol.py") < post_core_cmd.index(
        "generate_mrm_report.py"
    )
    assert "generate_dependency_summary.py" in post_core_cmd


def test_split_step_command_extracts_prelude_and_subcommands() -> None:
    prelude, subcommands = lp._split_step_command(
        "source .venv/bin/activate && uv run python scripts/a.py && uv run python scripts/b.py"
    )
    assert prelude == "source .venv/bin/activate"
    assert subcommands == ["uv run python scripts/a.py", "uv run python scripts/b.py"]


def test_build_steps_balanced_profile_applies_expected_sampling_mix() -> None:
    steps = lp.build_steps(
        "run-balanced",
        include_rapids=True,
        include_notebooks=False,
        sampling_profile="balanced",
    )
    by_name = {name: cmd for name, _required, cmd in steps}
    main_pre_cmd = by_name["main_pre"]
    heavy_main_cmd = by_name["heavy_main"]
    causal_cmd = by_name["causal"]
    cate_cmd = by_name["cate_portfolio"]
    rapids_cmd = by_name["rapids"]

    assert "materialize_feature_artifacts.py" in main_pre_cmd
    assert "--sample_size 0" in main_pre_cmd
    assert "run_survival_analysis.py --sample_size 250000 --rsf_n_estimators 200" in heavy_main_cmd
    assert "train_lgd_ead.py --sample_size 0" in heavy_main_cmd
    assert (
        "scripts.optimize_portfolio --config configs/optimization.yaml --max_candidates 20000"
        in (heavy_main_cmd)
    )
    assert (
        "scripts.optimize_portfolio_tradeoff --config configs/optimization.yaml --max_candidates 80000 --grid-profile quick"
        in heavy_main_cmd
    )
    assert (
        "scripts.select_economic_portfolio_policy --config configs/optimization.yaml --run-tag run-balanced"
        in heavy_main_cmd
    )
    assert (
        "scripts.simulate_ab_test --max_portfolio_pd 0.18 --max_candidates 20000 --n_boot 5000"
        in heavy_main_cmd
    )
    assert "--policy_selector explicit_champion_only" in heavy_main_cmd
    assert (
        "bash scripts/causal/run_in_causal_env.sh scripts/estimate_causal_effects.py --treatment int_rate --sample_size 200000 --run_tag run-balanced"
        in causal_cmd
    )
    assert (
        "bash scripts/causal/run_in_causal_env.sh scripts/simulate_causal_policy.py" in causal_cmd
    )
    assert (
        "bash scripts/causal/run_in_causal_env.sh scripts/validate_causal_policy.py" in causal_cmd
    )
    assert (
        "bash scripts/causal/run_in_causal_env.sh scripts/backtest_causal_policy_oot.py"
        in causal_cmd
    )
    assert "scripts.optimize_cate_portfolio --max_candidates 20000" in cate_cmd
    assert "--profile current" in rapids_cmd


def test_build_steps_smoke_profile_uses_small_resumability_args() -> None:
    steps = lp.build_steps(
        "run-smoke",
        include_rapids=False,
        include_notebooks=False,
        sampling_profile="smoke",
        profile_cfg={
            "search_space": {
                "pd": {"config_path": "configs/pd_model.smart.yaml"},
                "survival": {
                    "full_data": False,
                    "sample_size": 5000,
                    "rsf_sample_size": 2000,
                    "rsf_n_estimators": 10,
                    "rsf_max_samples": 0.25,
                    "rsf_n_jobs": 1,
                },
                "portfolio": {"max_candidates": 100},
                "tradeoff": {"max_candidates": 100, "grid_profile": "quick"},
                "ab": {
                    "max_portfolio_pd": 0.18,
                    "max_candidates": 100,
                    "n_boot": 100,
                    "seed": 42,
                    "no_regression_tolerance_pct": 0.05,
                },
                "causal": {"sample_size": 2000},
                "cate_portfolio": {"max_candidates": 100},
            }
        },
    )
    by_name = {name: cmd for name, _required, cmd in steps}
    assert "--sample_size 20000" in by_name["main_pre"]
    assert (
        "run_survival_analysis.py --sample_size 5000 --rsf_n_estimators 10 --rsf_sample_size 2000 "
        "--rsf_n_jobs 1 --rsf_max_samples 0.25" in by_name["heavy_main"]
    )
    assert "train_lgd_ead.py --sample_size 10000 --benchmark-short" in by_name["heavy_main"]
    assert (
        "scripts.optimize_portfolio --config configs/optimization.yaml --max_candidates 100"
        in by_name["heavy_main"]
    )
    assert (
        "scripts.optimize_portfolio_tradeoff --config configs/optimization.yaml --max_candidates 100 --grid-profile quick"
        in by_name["heavy_main"]
    )
    assert (
        "bash scripts/causal/run_in_causal_env.sh scripts/estimate_causal_effects.py --treatment int_rate --sample_size 2000 --run_tag run-smoke"
        in by_name["causal"]
    )


def test_build_steps_post_core_includes_explicit_comparison_baseline() -> None:
    steps = lp.build_steps(
        "run-baseline",
        include_rapids=False,
        include_notebooks=False,
        sampling_profile="balanced",
        comparison_baseline="/tmp/fixed_baseline_snapshot.json",
    )
    post_core_cmd = next(cmd for name, _required, cmd in steps if name == "post_core")
    assert (
        "run_comparison.py compare --run-tag run-baseline --baseline /tmp/fixed_baseline_snapshot.json"
        in post_core_cmd
    )


def test_build_steps_conformal_search_passes_partition_candidates_and_shrinkback() -> None:
    steps = lp.build_steps(
        "run-conf",
        include_rapids=False,
        include_notebooks=False,
        sampling_profile="smoke",
        profile_cfg={
            "search_space": {
                "pd": {"config_path": "configs/pd_model.smart.yaml"},
                "conformal": {
                    "partition_candidates": [
                        "score_decile_mondrian",
                        "grade",
                        "grade_x_scoreband_mondrian",
                    ],
                    "shrinkback_enabled": True,
                    "group_coverage_floor_enabled": True,
                    "scaled_scores_options": [True, False],
                    "group_multiplier_grid": [1.0, 1.01, 1.03],
                    "temporal_multiplier_grid": [1.0, 1.01, 1.03],
                },
            }
        },
    )
    by_name = {name: cmd for name, _required, cmd in steps}
    main_pre_cmd = by_name["main_pre"]
    assert (
        "--partition_candidates score_decile_mondrian,grade,grade_x_scoreband_mondrian"
        in main_pre_cmd
    )
    assert "--shrinkback_enabled 1" in main_pre_cmd
    assert "--group_coverage_floor_enabled 1" in main_pre_cmd
    assert "--scaled_scores_options True,False" in main_pre_cmd


def test_profile_default_comparison_baseline_run_tag_is_resolved(tmp_path, monkeypatch) -> None:
    comparisons = tmp_path / "reports" / "run_comparisons" / "baseline-tag"
    comparisons.mkdir(parents=True)
    (comparisons / "baseline_snapshot.json").write_text("{}", encoding="utf-8")

    profile_cfg = {
        "defaults": {
            "comparison_baseline_run_tag": "baseline-tag",
        }
    }
    monkeypatch.setattr(lp, "REPO_ROOT", tmp_path)

    resolved = lp._resolve_comparison_baseline_from_profile(profile_cfg)
    assert resolved == (comparisons / "baseline_snapshot.json").resolve()


def test_build_steps_mega64safe_caps_survival_memory_pressure() -> None:
    steps = lp.build_steps(
        "run-mega64safe",
        include_rapids=False,
        include_notebooks=False,
        sampling_profile="mega64safe",
    )
    by_name = {name: cmd for name, _required, cmd in steps}
    heavy_main_cmd = by_name["heavy_main"]

    assert (
        "run_survival_analysis.py --full-data --rsf_n_estimators 200 --rsf_sample_size 500000 "
        "--rsf_max_samples 0.5 --rsf_n_jobs 12" in heavy_main_cmd
    )
    assert (
        "scripts.optimize_portfolio --config configs/optimization.yaml --max_candidates 150000"
        in (heavy_main_cmd)
    )


def test_build_steps_champion64safe_keeps_search_phases_enabled(tmp_path, monkeypatch) -> None:
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

    steps = lp.build_steps(
        "run-champion64safe",
        include_rapids=False,
        include_notebooks=False,
        sampling_profile="champion64safe",
    )
    heavy_main_cmd = next(cmd for name, _required, cmd in steps if name == "heavy_main")

    assert "scripts.optimize_portfolio_tradeoff" in heavy_main_cmd
    assert "scripts.select_economic_portfolio_policy" in heavy_main_cmd
    assert "scripts.simulate_ab_test" in heavy_main_cmd
    assert "--policy_selector explicit_champion_only" in heavy_main_cmd


def test_build_steps_challenger_promotion_uses_economic_search_driver(
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
  actual_ab_top_k: 20
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

    steps = lp.build_steps(
        "run-monotonic",
        include_rapids=False,
        include_notebooks=False,
        sampling_profile="champion64safe",
        pipeline_family="challenger_promotion",
        profile_cfg={
            "search_space": {
                "portfolio": {"max_candidates": 150000},
                "tradeoff": {"max_candidates": 80000, "grid_profile": "balanced"},
                "ab": {
                    "execution_mode": "economic_search",
                    "max_portfolio_pd": 0.18,
                    "max_candidates": 150000,
                    "n_boot": 5000,
                    "seed": 42,
                    "policy_selector_default": "explicit_champion_only",
                    "policy_selector_candidates": [
                        "explicit_champion_only",
                        "actual_ab_guarded",
                    ],
                    "decision_scenarios": ["baseline", "selective_ambiguity_defer"],
                    "actual_ab_top_k": 20,
                },
            }
        },
    )
    heavy_main_cmd = next(cmd for name, _required, cmd in steps if name == "heavy_main")

    assert (
        "scripts.optimize_portfolio --config configs/optimization.yaml --max_candidates 150000"
        in heavy_main_cmd
    )
    assert "scripts/search_monotonic_economic_promotion.py" in heavy_main_cmd
    assert "--policy_selector_candidates explicit_champion_only,actual_ab_guarded" in heavy_main_cmd
    assert "--decision_scenarios baseline,selective_ambiguity_defer" in heavy_main_cmd
    assert "--tradeoff_max_candidates 80000" in heavy_main_cmd
    assert "--grid_profile balanced" in heavy_main_cmd
    assert "scripts.select_economic_portfolio_policy" not in heavy_main_cmd


def test_build_steps_canonical_confirmatory_full_uses_replay_and_full_policy_path(
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
    (repo / "configs" / "baselines").mkdir(parents=True, exist_ok=True)
    (repo / "configs" / "baselines" / "clean_baseline_manifest.json").write_text(
        "{}", encoding="utf-8"
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

    steps = lp.build_steps(
        "run-confirm",
        include_rapids=False,
        include_notebooks=False,
        sampling_profile="champion64safe",
        pipeline_family="canonical_rebuild",
        profile_cfg={
            "defaults": {
                "replay_manifest": "configs/baselines/clean_baseline_manifest.json",
                "pd_replay": True,
                "conformal_replay": True,
                "confirmatory_full": True,
            }
        },
    )
    by_name = {name: cmd for name, _required, cmd in steps}

    assert "--mode replay --replay_manifest" in by_name["preflight"]
    assert "--mode replay --replay_manifest" in by_name["main_pre"]
    heavy_main_cmd = by_name["heavy_main"]
    assert "scripts.optimize_portfolio_tradeoff" in heavy_main_cmd
    assert "scripts.select_economic_portfolio_policy" in heavy_main_cmd
    assert "scripts.simulate_ab_test" in heavy_main_cmd


def test_build_steps_notebooks_avoids_redundant_paper_suite_execution() -> None:
    steps = lp.build_steps("run-notebooks", include_rapids=False, include_notebooks=True)
    notebooks_cmd = next(cmd for name, _required, cmd in steps if name == "notebooks")
    assert "run_all_notebooks.py" in notebooks_cmd
    assert "extract_notebook_images.py" in notebooks_cmd
    assert "run_paper_notebook_suite.py" not in notebooks_cmd


def test_build_steps_canonical_rebuild_forces_frozen_policy_and_bundle(
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

    steps = lp.build_steps(
        "canonical-run",
        include_rapids=False,
        include_notebooks=False,
        sampling_profile="champion64safe",
        pipeline_family="canonical_rebuild",
    )
    by_name = {name: cmd for name, _required, cmd in steps}
    assert "--config configs/pd_model.champion.yaml" in by_name["main_pre"]
    assert "scripts.optimize_portfolio_tradeoff" not in by_name["heavy_main"]
    assert "scripts.select_economic_portfolio_policy" not in by_name["heavy_main"]
    assert "build_champion_search_bundle.py" in by_name["post_core"]


def test_preflight_validates_pd_config_before_long_run() -> None:
    steps = lp.build_steps("run-preflight", include_rapids=False, include_notebooks=False)
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

    steps = lp.build_steps(
        "champion-max",
        include_rapids=True,
        include_notebooks=True,
        sampling_profile="mega64plus",
        pipeline_family="champion_search",
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
                "pd": {"config_path": "configs/pd_model.champion_search_max.yaml"},
                "portfolio": {"max_candidates": 180000},
                "tradeoff": {"max_candidates": 150000, "grid_profile": "balanced"},
                "ab": {"max_portfolio_pd": 0.18, "max_candidates": 180000, "n_boot": 10000},
                "cate_portfolio": {"max_candidates": 180000},
                "rapids": {"ifrs9_mc": {"n_scenarios": 8192, "chunk_size": 256}},
            },
        },
    )

    by_name = {name: cmd for name, _required, cmd in steps}
    assert "rapids-python -u -m scripts.optimize_portfolio" in by_name["heavy_main"]
    assert "--solver_backend cuopt" in by_name["heavy_main"]
    assert "--catboost_backend gpu" in by_name["heavy_main"]
    assert "rapids-python -u -m scripts.optimize_cate_portfolio" in by_name["cate_portfolio"]
    assert "run_ifrs9_monte_carlo_gpu.py --n-scenarios 8192 --chunk-size 256" in by_name["rapids"]


def test_main_rejects_core_run_without_explicit_baseline(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(lp, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(lp, "BASELINE_REGISTRY_PATH", tmp_path / "missing_baseline_registry.json")
    monkeypatch.setattr(
        lp,
        "parse_args",
        lambda: argparse.Namespace(
            run_tag="2026-03-04-C-core-balanced",
            resume=False,
            refresh_baseline_on_resume=True,
            env_file=None,
            no_rapids=True,
            no_notebooks=True,
            stop_on_optional_failure=False,
            stall_window_minutes=15,
            from_step=None,
            until_step=None,
            sampling_profile="balanced",
            comparison_baseline=None,
            comparison_baseline_run_tag=None,
        ),
    )
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
    registry_path.write_text(
        json.dumps({"official_run_tag": baseline_run_tag}),
        encoding="utf-8",
    )
    monkeypatch.setattr(lp, "BASELINE_REGISTRY_PATH", registry_path)
    monkeypatch.setattr(
        lp,
        "parse_args",
        lambda: argparse.Namespace(
            run_tag="2026-03-04-C-core-balanced",
            resume=False,
            refresh_baseline_on_resume=True,
            env_file=None,
            no_rapids=True,
            no_notebooks=True,
            stop_on_optional_failure=False,
            stall_window_minutes=15,
            from_step=None,
            until_step=None,
            sampling_profile="balanced",
            comparison_baseline=None,
            comparison_baseline_run_tag=None,
        ),
    )
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
        lp,
        "parse_args",
        lambda: argparse.Namespace(
            run_tag="test-run-stop",
            resume=False,
            refresh_baseline_on_resume=True,
            env_file=None,
            no_rapids=True,
            no_notebooks=True,
            stop_on_optional_failure=True,
            stall_window_minutes=15,
            from_step=None,
            until_step=None,
            sampling_profile="full",
            comparison_baseline=None,
            comparison_baseline_run_tag=None,
        ),
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
    ) -> int:
        _ = step_eta_default_seconds, stall_window_seconds, resume_subphases
        if required:
            return 0
        if step == "heavy_main":
            return 2
        return 0

    monkeypatch.setattr(lp, "run_step", fake_run_step)

    exit_code = lp.main()
    summary_path = tmp_path / "reports" / "run_logs" / "test-run-stop" / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert exit_code == 1
    assert summary["final_exit_code"] == 1
    assert summary["failed_required"] is False
    assert summary["stopped_on_optional_failure"] is True
    assert summary["failed_steps"] == ["heavy_main"]


def test_main_optional_failure_without_stop_flag_keeps_zero_exit(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(lp, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        lp,
        "parse_args",
        lambda: argparse.Namespace(
            run_tag="test-run-continue",
            resume=False,
            refresh_baseline_on_resume=True,
            env_file=None,
            no_rapids=True,
            no_notebooks=True,
            stop_on_optional_failure=False,
            stall_window_minutes=15,
            from_step=None,
            until_step=None,
            sampling_profile="full",
            comparison_baseline=None,
            comparison_baseline_run_tag=None,
        ),
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
    ) -> int:
        _ = step_eta_default_seconds, stall_window_seconds, resume_subphases
        if required:
            return 0
        if step == "heavy_main":
            return 2
        return 0

    monkeypatch.setattr(lp, "run_step", fake_run_step)

    exit_code = lp.main()
    summary_path = tmp_path / "reports" / "run_logs" / "test-run-continue" / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert summary["final_exit_code"] == 0
    assert summary["stopped_on_optional_failure"] is False
    assert summary["failed_steps"] == ["heavy_main"]


def test_main_runs_selected_step_window_only(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(lp, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        lp,
        "parse_args",
        lambda: argparse.Namespace(
            run_tag="test-run-window",
            resume=False,
            refresh_baseline_on_resume=True,
            env_file=None,
            no_rapids=True,
            no_notebooks=True,
            stop_on_optional_failure=False,
            stall_window_minutes=15,
            from_step="causal",
            until_step="post_core",
            sampling_profile="full",
            comparison_baseline=None,
            comparison_baseline_run_tag=None,
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
    ) -> int:
        _ = required, step_eta_default_seconds, stall_window_seconds, resume_subphases
        seen_steps.append(step)
        return 0

    monkeypatch.setattr(lp, "run_step", fake_run_step)

    exit_code = lp.main()

    assert exit_code == 0
    assert seen_steps == ["causal", "cate_portfolio", "post_core"]


def test_resume_refreshes_baseline_snapshot_when_preflight_skipped(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(lp, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        lp,
        "parse_args",
        lambda: argparse.Namespace(
            run_tag="test-run-resume-refresh",
            resume=True,
            refresh_baseline_on_resume=True,
            env_file=None,
            no_rapids=True,
            no_notebooks=True,
            stop_on_optional_failure=False,
            stall_window_minutes=15,
            from_step=None,
            until_step=None,
            sampling_profile="full",
            comparison_baseline=None,
            comparison_baseline_run_tag=None,
        ),
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
    ) -> int:
        _ = required, step_eta_default_seconds, stall_window_seconds, resume_subphases
        seen_steps.append(step)
        return 0

    monkeypatch.setattr(lp, "run_step", fake_run_step)
    exit_code = lp.main()

    assert exit_code == 0
    assert refreshed["value"] is True
    assert "preflight" not in seen_steps


def test_resume_skips_refresh_when_external_comparison_baseline_set(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(lp, "REPO_ROOT", tmp_path)
    baseline_path = (
        tmp_path / "reports" / "run_comparisons" / "fixed-baseline" / "baseline_snapshot.json"
    )
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        lp,
        "parse_args",
        lambda: argparse.Namespace(
            run_tag="test-run-resume-external-baseline",
            resume=True,
            refresh_baseline_on_resume=True,
            env_file=None,
            no_rapids=True,
            no_notebooks=True,
            stop_on_optional_failure=False,
            stall_window_minutes=15,
            from_step=None,
            until_step=None,
            sampling_profile="full",
            comparison_baseline=None,
            comparison_baseline_run_tag="fixed-baseline",
        ),
    )

    def _completed(_run_tag: str, step: str) -> bool:
        return step == "preflight"

    monkeypatch.setattr(lp, "load_completed_ok", _completed)
    refreshed = {"value": False}

    def _refresh(_run_tag: str) -> bool:
        refreshed["value"] = True
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
    ) -> int:
        _ = required, step_eta_default_seconds, stall_window_seconds, resume_subphases
        seen_steps.append(step)
        return 0

    monkeypatch.setattr(lp, "run_step", fake_run_step)
    exit_code = lp.main()

    assert exit_code == 0
    assert refreshed["value"] is False
    assert "preflight" not in seen_steps
