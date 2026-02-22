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


# ── MRM Policy Config ──

MRM_CONFIG_PATH = PROJECT_ROOT / "configs" / "mrm_policy.yaml"


class TestMRMConfig:
    """Validate MRM policy configuration structure."""

    @pytest.fixture(autouse=True)
    def _load_mrm(self):
        if not MRM_CONFIG_PATH.exists():
            pytest.skip("MRM config not found")
        with open(MRM_CONFIG_PATH) as f:
            self.cfg = yaml.safe_load(f)

    def test_mrm_config_required_keys(self):
        """MRM config must have all required top-level sections."""
        required = {"model", "governance", "retraining_triggers", "challenger", "output"}
        assert required.issubset(self.cfg.keys()), (
            f"Missing keys: {required - set(self.cfg.keys())}"
        )

    def test_mrm_retraining_triggers_are_positive(self):
        """All retraining trigger thresholds must be positive."""
        triggers = self.cfg["retraining_triggers"]
        for key, value in triggers.items():
            assert value > 0, f"Trigger '{key}' must be positive, got {value}"

    def test_mrm_champion_artifact_path_is_posix(self):
        """Champion artifact path should use forward slashes."""
        path = self.cfg["model"]["champion_artifact"]
        assert "\\" not in path, f"Path should use forward slashes: {path}"


# ── Conformal Policy Config ──

CONFORMAL_POLICY_PATH = PROJECT_ROOT / "configs" / "conformal_policy.yaml"


class TestConformalPolicyConfig:
    """Validate conformal policy config invariants."""

    @pytest.fixture(autouse=True)
    def _load(self):
        if not CONFORMAL_POLICY_PATH.exists():
            pytest.skip("Conformal policy config not found")
        with open(CONFORMAL_POLICY_PATH) as f:
            self.cfg = yaml.safe_load(f)
        self.policy = self.cfg["policy"]

    def test_required_sections(self):
        assert {"policy", "artifacts", "output"}.issubset(self.cfg.keys())

    def test_coverage_targets_in_valid_range(self):
        for key in ("target_coverage_90_min", "target_coverage_95_min"):
            val = self.policy[key]
            assert 0.5 < val < 1.0, f"{key}={val} outside (0.5, 1.0)"

    def test_95_target_exceeds_90_target(self):
        assert self.policy["target_coverage_95_min"] > self.policy["target_coverage_90_min"]

    def test_group_floor_below_global_target(self):
        assert self.policy["min_group_coverage_90_min"] <= self.policy["target_coverage_90_min"]

    def test_width_budget_is_positive(self):
        assert 0.0 < self.policy["max_avg_width_90"] <= 2.0

    def test_alert_thresholds_non_negative(self):
        for key in ("max_critical_alerts", "max_total_alerts", "max_warning_alerts"):
            assert self.policy[key] >= 0, f"{key} must be >= 0"

    def test_critical_alerts_leq_total(self):
        assert self.policy["max_critical_alerts"] <= self.policy["max_total_alerts"]

    def test_artifact_paths_have_valid_extensions(self):
        for key, path in self.cfg["artifacts"].items():
            assert path.endswith((".pkl", ".parquet", ".json")), (
                f"Artifact '{key}' has unexpected extension: {path}"
            )


# ── Fairness Policy Config ──

FAIRNESS_POLICY_PATH = PROJECT_ROOT / "configs" / "fairness_policy.yaml"


class TestFairnessPolicyConfig:
    """Validate fairness policy config invariants."""

    @pytest.fixture(autouse=True)
    def _load(self):
        if not FAIRNESS_POLICY_PATH.exists():
            pytest.skip("Fairness policy config not found")
        with open(FAIRNESS_POLICY_PATH) as f:
            self.cfg = yaml.safe_load(f)
        self.policy = self.cfg["policy"]

    def test_required_sections(self):
        assert {"policy", "attributes", "artifacts", "output"}.issubset(self.cfg.keys())

    def test_dpd_threshold_range(self):
        val = self.policy["dpd_threshold"]
        assert 0 < val <= 0.5, f"dpd_threshold={val} outside (0, 0.5]"

    def test_eo_gap_threshold_range(self):
        val = self.policy["eo_gap_threshold"]
        assert 0 < val <= 0.5, f"eo_gap_threshold={val} outside (0, 0.5]"

    def test_dir_threshold_range(self):
        val = self.policy["dir_threshold"]
        assert 0.5 <= val < 1.0, f"dir_threshold={val} outside [0.5, 1.0)"

    def test_prediction_threshold_is_probability(self):
        val = self.policy["prediction_threshold"]
        assert 0 < val < 1.0, f"prediction_threshold={val} outside (0, 1)"

    def test_at_least_one_attribute(self):
        assert len(self.cfg["attributes"]) >= 1

    def test_attributes_have_name_and_column(self):
        for attr in self.cfg["attributes"]:
            assert "name" in attr, f"Attribute missing 'name': {attr}"
            assert "column" in attr, f"Attribute missing 'column': {attr}"

    def test_binning_enum_if_present(self):
        valid = {"quartile", "decile"}
        for attr in self.cfg["attributes"]:
            if "binning" in attr:
                assert attr["binning"] in valid, (
                    f"Attribute '{attr['name']}' has invalid binning: {attr['binning']}"
                )

    def test_output_paths_have_valid_extensions(self):
        for key, path in self.cfg["output"].items():
            assert path.endswith((".parquet", ".json")), (
                f"Output '{key}' has unexpected extension: {path}"
            )


# ── Optimization Config ──

OPTIMIZATION_CONFIG_PATH = PROJECT_ROOT / "configs" / "optimization.yaml"


class TestOptimizationConfig:
    """Validate portfolio optimization config invariants."""

    @pytest.fixture(autouse=True)
    def _load(self):
        if not OPTIMIZATION_CONFIG_PATH.exists():
            pytest.skip("Optimization config not found")
        with open(OPTIMIZATION_CONFIG_PATH) as f:
            self.cfg = yaml.safe_load(f)

    def test_required_sections(self):
        assert {"optimization", "portfolio", "conformal"}.issubset(self.cfg.keys())

    def test_solver_type_valid(self):
        assert self.cfg["optimization"]["type"] in {"LP", "MILP"}

    def test_solver_name_valid(self):
        assert self.cfg["optimization"]["solver"] in {"highs", "cplex", "gurobi", "glpk"}

    def test_time_limit_reasonable(self):
        val = self.cfg["optimization"]["time_limit"]
        assert 10 <= val <= 3600, f"time_limit={val} outside [10, 3600]"

    def test_threads_positive(self):
        assert self.cfg["optimization"]["threads"] >= 1

    def test_budget_positive(self):
        assert self.cfg["portfolio"]["total_budget"] > 0

    def test_max_concentration_range(self):
        val = self.cfg["portfolio"]["max_concentration"]
        assert 0 < val <= 1.0, f"max_concentration={val} outside (0, 1]"

    def test_max_portfolio_pd_range(self):
        val = self.cfg["portfolio"]["max_portfolio_pd"]
        assert 0 < val < 0.5, f"max_portfolio_pd={val} outside (0, 0.5)"

    def test_conformal_alpha_valid(self):
        val = self.cfg["conformal"]["alpha"]
        assert val in {0.05, 0.10, 0.20}, f"alpha={val} not in supported set"
