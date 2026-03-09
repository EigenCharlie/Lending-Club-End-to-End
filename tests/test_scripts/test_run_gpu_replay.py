from __future__ import annotations

from scripts.run_gpu_replay import build_stage_commands


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
