from __future__ import annotations

from scripts.run_gpu_replay import build_post_replay_commands, build_stage_commands


def test_build_stage_commands_mega64plus_uses_gpu_backends() -> None:
    commands = build_stage_commands(
        run_tag="gpu-run-x",
        profile="mega64plus",
        pd_config="configs/pd_model.gpu.yaml",
        optimization_config="configs/optimization.yaml",
    )

    assert "--config configs/pd_model.gpu.yaml --sample_size 0" in commands["pd"]
    assert "--catboost_backend gpu" in commands["lgd_ead"]
    assert "--max_candidates 150000 --solver_backend cuopt" in commands["portfolio"]
    assert (
        "--max_candidates 80000 --grid-profile night --solver_backend cuopt" in commands["tradeoff"]
    )
    assert "--max_candidates 150000" in commands["ab"]
    assert "--solver_backend cuopt" in commands["ab"]
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
