"""Tests for config/code consistency.

Validates that configuration files match actual code usage patterns,
preventing the class of bug where configs drift from implementation.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "pd_model.yaml"
CONTRACT_PATH = PROJECT_ROOT / "models" / "pd_model_contract.json"
DVC_YAML_PATH = PROJECT_ROOT / "dvc.yaml"
DVC_LOCK_PATH = PROJECT_ROOT / "dvc.lock"


@pytest.fixture
def pd_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


class TestCalibrationConfig:
    def test_calibration_method_is_valid(self, pd_config: dict) -> None:
        method = pd_config["calibration"]["method"]
        assert method in {"isotonic", "platt", "venn_abers"}, (
            f"Unknown calibration method: {method}"
        )

    def test_calibration_method_matches_thesis(self, pd_config: dict) -> None:
        """Thesis selected Platt sigmoid (ECE=0.0128). Config must agree."""
        method = pd_config["calibration"]["method"]
        assert method == "platt", (
            f"Config says '{method}' but thesis/CLAUDE.md specifies Platt sigmoid"
        )


class TestConformalConfig:
    def test_uses_mapie_13_params(self, pd_config: dict) -> None:
        """Config should use MAPIE 1.3+ parameter names."""
        conf = pd_config.get("conformal", {})
        legacy_keys = {"alpha", "method", "cv"}
        found_legacy = legacy_keys & set(conf.keys())
        assert not found_legacy, (
            f"Found legacy MAPIE <1.0 params: {found_legacy}. "
            "Use confidence_level/prefit/estimator instead."
        )

    def test_confidence_level_valid(self, pd_config: dict) -> None:
        conf = pd_config.get("conformal", {})
        level = conf.get("confidence_level", 0.9)
        assert 0.5 < level < 1.0, f"confidence_level={level} outside valid range (0.5, 1.0)"


class TestModelContract:
    @pytest.mark.skipif(not CONTRACT_PATH.exists(), reason="No model contract found")
    def test_paths_are_posix(self) -> None:
        """Model contract paths must not contain Windows backslashes."""
        contract = json.loads(CONTRACT_PATH.read_text())
        for key in ("model_path", "calibrator_path"):
            path = contract.get(key, "")
            if path:
                assert "\\" not in path, (
                    f"Contract {key}='{path}' contains Windows backslashes. "
                    "Use Path.as_posix() in pd_contract.py."
                )

    @pytest.mark.skipif(not CONTRACT_PATH.exists(), reason="No model contract found")
    def test_contract_features_not_empty(self) -> None:
        contract = json.loads(CONTRACT_PATH.read_text())
        features = contract.get("feature_names", [])
        assert len(features) > 0, "Model contract has no features"

    @pytest.mark.skipif(not CONTRACT_PATH.exists(), reason="No model contract found")
    def test_no_missing_features_in_splits(self) -> None:
        contract = json.loads(CONTRACT_PATH.read_text())
        missing = contract.get("split_missing_features", {})
        for split_name, missing_list in missing.items():
            assert len(missing_list) == 0, (
                f"Split '{split_name}' is missing features: {missing_list}"
            )


class TestDvcPipeline:
    """Validate dvc.yaml DAG consistency with actual project files."""

    @pytest.fixture
    def dvc_config(self) -> dict:
        return yaml.safe_load(DVC_YAML_PATH.read_text())

    @pytest.fixture
    def dvc_lock(self) -> dict:
        return yaml.safe_load(DVC_LOCK_PATH.read_text())

    @pytest.mark.skipif(not DVC_YAML_PATH.exists(), reason="No dvc.yaml found")
    def test_all_stage_scripts_exist(self, dvc_config: dict) -> None:
        """Every cmd in dvc.yaml should reference scripts/modules that exist."""
        stages = dvc_config.get("stages", {})
        for name, stage in stages.items():
            cmd = stage.get("cmd", "")
            # Extract script path from cmd patterns like:
            # "uv run python scripts/foo.py" or "uv run python -c \"from src.x import ...\""
            if "scripts/" in cmd:
                script = cmd.split("scripts/")[1].split()[0].split("--")[0].strip()
                script_path = PROJECT_ROOT / "scripts" / script
                assert script_path.exists(), (
                    f"Stage '{name}' references scripts/{script} but file doesn't exist"
                )
            elif "from src." in cmd:
                # e.g. "from src.data.make_dataset import main"
                module_str = cmd.split("from ")[1].split(" import")[0]
                module_path = PROJECT_ROOT / (module_str.replace(".", "/") + ".py")
                assert module_path.exists(), (
                    f"Stage '{name}' references {module_str} but file doesn't exist"
                )

    @pytest.mark.skipif(not DVC_YAML_PATH.exists(), reason="No dvc.yaml found")
    def test_config_deps_exist(self, dvc_config: dict) -> None:
        """Config file dependencies (configs/*.yaml) should exist on disk."""
        stages = dvc_config.get("stages", {})
        for name, stage in stages.items():
            for dep in stage.get("deps", []):
                if dep.startswith("configs/"):
                    dep_path = PROJECT_ROOT / dep
                    assert dep_path.exists(), (
                        f"Stage '{name}' depends on {dep} but file doesn't exist"
                    )

    @pytest.mark.skipif(not DVC_YAML_PATH.exists(), reason="No dvc.yaml found")
    def test_no_overlapping_outputs(self, dvc_config: dict) -> None:
        """No two stages should declare the same output file."""
        stages = dvc_config.get("stages", {})
        seen: dict[str, str] = {}
        for name, stage in stages.items():
            for out in stage.get("outs", []):
                assert out not in seen, (
                    f"Output '{out}' declared in both '{seen[out]}' and '{name}'"
                )
                seen[out] = name

    @pytest.mark.skipif(not DVC_YAML_PATH.exists(), reason="No dvc.yaml found")
    def test_expected_showcase_stages_present(self, dvc_config: dict) -> None:
        """DVC DAG should include governance/export stages used by Streamlit + MLflow."""
        stages = set(dvc_config.get("stages", {}).keys())
        required = {
            "backtest_conformal_coverage",
            "validate_conformal_policy",
            "export_streamlit_artifacts",
            "export_storytelling_snapshot",
        }
        missing = required - stages
        assert not missing, f"Missing DVC showcase stages: {sorted(missing)}"

    @pytest.mark.skipif(not DVC_LOCK_PATH.exists(), reason="No dvc.lock found")
    def test_dvc_yaml_outs_have_lock_metadata(self, dvc_config: dict, dvc_lock: dict) -> None:
        """Each output declared in dvc.yaml should be versioned in dvc.lock."""
        yaml_stages = dvc_config.get("stages", {})
        lock_stages = dvc_lock.get("stages", {})

        missing_stages = sorted(set(yaml_stages) - set(lock_stages))
        assert not missing_stages, (
            f"Stages present in dvc.yaml but missing in dvc.lock: {missing_stages}"
        )

        for stage_name, stage_cfg in yaml_stages.items():
            declared_outs = stage_cfg.get("outs", [])
            lock_outs = {out.get("path") for out in lock_stages.get(stage_name, {}).get("outs", [])}
            missing_outs = [out for out in declared_outs if out not in lock_outs]
            assert not missing_outs, (
                f"Stage '{stage_name}' has outputs without lock metadata: {missing_outs}. "
                "Run `dvc repro` for affected stages."
            )


class TestGitDvcHygiene:
    @pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
    def test_dvc_json_outputs_not_tracked_and_ignored(self) -> None:
        outputs = [
            "models/conformal_policy_status.json",
            "models/causal_policy_rule.json",
        ]

        tracked = set(subprocess.check_output(["git", "ls-files"], text=True).splitlines())
        for output in outputs:
            assert output not in tracked, f"{output} should be DVC-managed, not git-tracked"

            ignored = subprocess.run(["git", "check-ignore", "-q", output], check=False)
            assert ignored.returncode == 0, (
                f"{output} should be ignored by git to avoid git-vs-dvc tracking conflicts"
            )
