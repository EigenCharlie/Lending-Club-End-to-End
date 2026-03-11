from __future__ import annotations

from scripts.run_gpu_replay import (
    build_post_replay_commands,
    build_stage_commands,
    validate_rapids_env,
)


def test_build_stage_commands_mega64plus_uses_gpu_backends() -> None:
    commands = build_stage_commands(
        run_tag="gpu-run-x",
        profile="mega64plus",
        pd_config="configs/pd_model.gpu.yaml",
        optimization_config="configs/optimization.yaml",
    )

    assert "--config configs/pd_model.gpu.yaml --sample_size 0" in commands["pd"]
    assert "--catboost_backend gpu" in commands["lgd_ead"]
    assert commands["portfolio"].startswith("python -u -m scripts.optimize_portfolio")
    assert "--max_candidates 150000 --solver_backend cuopt" in commands["portfolio"]
    assert (
        "--max_candidates 80000 --grid-profile night --solver_backend cuopt" in commands["tradeoff"]
    )
    assert commands["policy_selection"].startswith(
        "python -u -m scripts.select_economic_portfolio_policy"
    )
    assert "--run-tag gpu-run-x --solver_backend cuopt" in commands["policy_selection"]
    assert "--max_candidates 150000" in commands["ab"]
    assert "--solver_backend cuopt" in commands["ab"]
    assert "--policy_selector explicit_champion_only" in commands["ab"]
    assert "--max_candidates 150000 --solver_backend cuopt" in commands["cate_portfolio"]


def test_build_post_replay_commands_can_chain_notebooks_and_images() -> None:
    commands = build_post_replay_commands(
        notebook_timeout=3600,
        notebook_output_dir="reports/notebook_exec",
        notebook_inplace=True,
        include_side_projects=True,
        extract_images_after=True,
    )

    assert commands[0][0] == "notebooks"
    assert (
        "--execute-all --include-side-projects --timeout 3600 --inplace true --output-dir reports/notebook_exec"
        in commands[0][1]
    )
    assert commands[1][0] == "extract_images"
    assert "--notebook-dir reports/notebook_exec/notebooks" in commands[1][1]


def test_validate_rapids_env_skips_when_no_rapids_stages() -> None:
    result = validate_rapids_env(selected_stages=["pd", "lgd_ead"])
    assert result["checked"] is False
    assert result["needs_rapids"] is False


def test_build_stage_commands_rapids_final_includes_ifrs9_mc() -> None:
    commands = build_stage_commands(
        run_tag="gpu-run-x",
        profile="rapids_final",
        pd_config="configs/pd_model.gpu.yaml",
        optimization_config="configs/optimization.yaml",
    )
    assert commands["portfolio"].endswith("--max_candidates 0 --solver_backend cuopt")
    assert (
        "--max_candidates 150000 --grid-profile night --solver_backend cuopt"
        in commands["tradeoff"]
    )
    assert "--max_candidates 0 --solver_backend cuopt" in commands["cate_portfolio"]
    assert commands["ifrs9_mc"].startswith("python -u scripts/run_ifrs9_monte_carlo_gpu.py")
    assert "--n-scenarios 8192 --chunk-size 256" in commands["ifrs9_mc"]
