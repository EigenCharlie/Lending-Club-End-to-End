"""Build a namespaced local-HPO PD config from the best monotonic search result."""

from __future__ import annotations

import argparse
import copy
import pickle
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_FEATURE_ALIASES = {
    "days_since_last_delinq": "delinq_recency",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return dict(payload) if isinstance(payload, dict) else {}


def _load_best_variant(base_search_run_tag: str) -> dict[str, Any]:
    base_dir = REPO_ROOT / "models" / "search_pd" / base_search_run_tag
    best_path = base_dir / "monotonic_competitor_best_blockwise_exhaustive.json"
    if best_path.exists():
        import json

        return dict(json.loads(best_path.read_text(encoding="utf-8")) or {})
    search_path = base_dir / "monotonic_competitor_search_blockwise_exhaustive.json"
    if search_path.exists():
        import json

        payload = dict(json.loads(search_path.read_text(encoding="utf-8")) or {})
        best = dict(payload.get("best_variant") or {})
        if best:
            return best
    raise FileNotFoundError(
        "No blockwise monotonic search result found for run tag "
        f"{base_search_run_tag!r} under {base_dir}"
    )


def _load_valid_catboost_features(
    feature_config_path: str | None,
    *,
    excluded_features: set[str] | None = None,
) -> set[str]:
    if not feature_config_path:
        return set()
    path = REPO_ROOT / feature_config_path
    if not path.exists():
        return set()
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        return set()
    features = payload.get("CATBOOST_FEATURES")
    if not isinstance(features, list):
        return set()
    valid = {str(feature).strip() for feature in features if str(feature).strip()}
    if excluded_features:
        valid -= {str(feature).strip() for feature in excluded_features if str(feature).strip()}
    return valid


def _canonical_feature_name(feature_name: str, valid_features: set[str]) -> str | None:
    candidate = str(feature_name).strip()
    if not candidate:
        return None
    aliased = LEGACY_FEATURE_ALIASES.get(candidate, candidate)
    if aliased in valid_features:
        return aliased
    if candidate in valid_features:
        return candidate
    return None


def _sanitize_local_refine_space(
    local_refine: dict[str, Any],
    *,
    valid_features: set[str],
) -> dict[str, Any]:
    if not valid_features:
        return copy.deepcopy(local_refine)

    sanitized = copy.deepcopy(local_refine)
    for section_name in ("first_feature_use_penalties", "feature_weights"):
        raw_section = dict(sanitized.get(section_name) or {})
        filtered_section: dict[str, Any] = {}
        for feature_name, spec in raw_section.items():
            canonical_name = _canonical_feature_name(feature_name, valid_features)
            if canonical_name is None:
                continue
            filtered_section[canonical_name] = spec
        if filtered_section:
            sanitized[section_name] = filtered_section
        else:
            sanitized.pop(section_name, None)
    return sanitized


def _local_refine_space(profile: str) -> dict[str, Any]:
    common = {
        "enqueue_base_trial": True,
        "fixed_params": {
            "bootstrap_type": "MVS",
            "grow_policy": "SymmetricTree",
        },
        "iterations": {"low": 2800, "high": 4200, "step": 50},
        "learning_rate": {"low": 0.015, "high": 0.06, "log": True},
        "depth": {"choices": [8, 9, 10]},
        "l2_leaf_reg": {"low": 30.0, "high": 260.0, "log": True},
        "min_data_in_leaf": {"low": 120, "high": 260, "step": 5},
        "rsm": {"low": 0.55, "high": 0.85},
        "random_strength": {"low": 1e-8, "high": 1e-3, "log": True},
        "border_count": {"choices": [128, 148, 192, 254]},
        "subsample": {"low": 0.65, "high": 0.9},
        "leaf_estimation_iterations": {"choices": [2, 3, 4, 5]},
        "penalties_coefficient": [0.75, 1.0, 1.25, 1.5],
        "first_feature_use_penalties": {
            "rev_utilization": [0.0, 0.2, 0.5],
            "days_since_last_delinq": [0.0, 0.2, 0.5],
        },
    }
    if profile == "blockwise_affordability":
        common["feature_weights"] = {
            "loan_to_income": [1.0, 1.15, 1.3],
            "annual_inc": [1.0, 1.1, 1.2],
            "dti": [1.0, 1.1, 1.2],
            "installment": [1.0, 1.05, 1.1],
        }
    elif profile == "blockwise_capacity":
        common["feature_weights"] = {
            "annual_inc": [1.0, 1.1, 1.2],
            "loan_amnt": [1.0, 1.05, 1.1],
            "installment": [1.0, 1.05, 1.1],
        }
    else:
        common["feature_weights"] = {
            "annual_inc": [1.0, 1.1],
            "loan_to_income": [1.0, 1.1],
        }
    return common


def build_pd_hpo_local_config(
    *,
    run_tag: str,
    base_search_run_tag: str,
    base_config_path: str = "configs/pd_model.champion.yaml",
    output_path: str | None = None,
    n_trials: int = 120,
) -> Path:
    base_config = _load_yaml(REPO_ROOT / base_config_path)
    best = _load_best_variant(base_search_run_tag)
    resolved_run_tag = str(run_tag).strip() or "pd-hpo-local-untracked"
    out_path = (
        REPO_ROOT / output_path
        if output_path
        else REPO_ROOT / "models" / "search_pd" / resolved_run_tag / "pd_model_hpo_local.yaml"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    config = copy.deepcopy(base_config)
    params = dict(best.get("params_reference") or {})
    config.setdefault("model", {})
    config["model"]["params"] = params
    feature_source_cfg = dict(config.get("feature_source") or {})
    stable_core_cfg = dict(config.get("stable_core") or {})
    excluded_features = set(stable_core_cfg.get("exclude_features") or [])
    valid_features = _load_valid_catboost_features(
        feature_source_cfg.get("feature_config_path"),
        excluded_features=excluded_features,
    )

    best_calibration = str(best.get("calibration_method", "venn_abers")).strip() or "venn_abers"
    config.setdefault("calibration", {})
    config["calibration"]["method"] = best_calibration
    config["calibration"]["candidates"] = [best_calibration]

    config.setdefault("hpo", {})
    config["hpo"].update(
        {
            "enabled": True,
            "n_trials": int(n_trials),
            "sampler": "tpe",
            "pruner": "median",
            "timeout_minutes": 0,
            "n_startup_trials": 30,
            "multivariate_tpe": True,
            "group_tpe": True,
            "warn_independent_sampling": False,
            "constant_liar": False,
            "pruner_n_startup_trials": 20,
            "pruner_n_warmup_steps": 75,
            "use_pruning_callback": True,
            "study_storage": f"sqlite:///models/search_pd/{resolved_run_tag}/optuna_pd_hpo_local.db",
            "study_name": f"pd_hpo_local_{best.get('profile', 'candidate')}_{best.get('variant_id', 'base')}",
            "load_if_exists": True,
            "refit_full_train": True,
            "gc_after_trial": True,
            "storage_heartbeat_interval": 60,
            "storage_grace_period": 240,
            "sqlite_timeout_seconds": 120,
            "retry_failed_trials": 2,
            "n_jobs": 1,
            "search_space_mode": "local_refine",
            "search_space_version": "cb_local_refine_v1",
            "local_refine": _sanitize_local_refine_space(
                _local_refine_space(str(best.get("profile", ""))),
                valid_features=valid_features,
            ),
            "constraints_policy": {
                "max_brier_delta": 0.0010,
                "max_ece_delta": 0.0025,
                "min_auc_delta": -0.0010,
            },
        }
    )

    config.setdefault("validation", {})
    seed_replay_cfg = dict(config["validation"].get("seed_replay", {}) or {})
    seed_replay_cfg.update(
        {
            "enabled": True,
            "top_k_trials": 3,
            "seeds": [42, 59, 76],
            "prioritize_gate_pass": True,
        }
    )
    config["validation"]["seed_replay"] = seed_replay_cfg

    namespaced_root = f"models/search_pd/{resolved_run_tag}"
    namespaced_data_root = f"data/processed/search_pd/{resolved_run_tag}"
    namespaced_fig_root = f"reports/figures/search_pd/{resolved_run_tag}"
    output_cfg = dict(config.get("output", {}) or {})
    output_cfg.update(
        {
            "model_path": f"{namespaced_root}/pd_local_hpo_tuned.cbm",
            "default_model_path": f"{namespaced_root}/pd_local_hpo_default.cbm",
            "tuned_model_path": f"{namespaced_root}/pd_local_hpo_tuned_alias.cbm",
            "conformal_path": f"{namespaced_root}/pd_local_hpo_calibrator.pkl",
            "status_path": f"{namespaced_root}/pd_training_status.json",
            "checkpoint_dir": f"{namespaced_root}/pd_training_checkpoints",
            "brier_decomposition_path": f"{namespaced_data_root}/brier_score_decomposition.json",
            "murphy_diagram_path": f"{namespaced_fig_root}/murphy_diagram.png",
            "canonical_model_path": f"{namespaced_root}/pd_candidate_model.cbm",
            "canonical_calibrator_path": f"{namespaced_root}/pd_candidate_calibrator.pkl",
            "contract_path": f"{namespaced_root}/pd_model_contract.json",
            "logreg_model_path": f"{namespaced_root}/pd_logreg_baseline.pkl",
            "training_record_path": f"{namespaced_root}/pd_training_record.pkl",
            "seed_replay_status_path": f"{namespaced_root}/pd_hpo_seed_replay_status.json",
            "test_predictions_path": f"{namespaced_data_root}/test_predictions.parquet",
            "shap_dir": f"{namespaced_fig_root}/shap",
            "threshold_semantics_path": f"{namespaced_root}/threshold_semantics.json",
        }
    )
    config["output"] = output_cfg

    decision_cfg = dict(config.get("decision_threshold", {}) or {})
    decision_cfg.update(
        {
            "output_path": f"{namespaced_root}/decision_threshold.json",
            "output_path_v2": f"{namespaced_root}/decision_threshold_v2.json",
        }
    )
    config["decision_threshold"] = decision_cfg

    config.setdefault("metadata", {})
    config["metadata"].update(
        {
            "run_tag": resolved_run_tag,
            "base_search_run_tag": base_search_run_tag,
            "base_config_path": base_config_path,
            "best_search_profile": best.get("profile"),
            "best_search_variant_id": best.get("variant_id"),
            "best_search_calibration_method": best_calibration,
            "best_search_threshold_mean": best.get("selected_threshold_mean"),
            "feature_prior_valid_count": len(valid_features),
        }
    )

    with out_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build PD local HPO config from best search result."
    )
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--base-search-run-tag", required=True)
    parser.add_argument("--base-config-path", default="configs/pd_model.champion.yaml")
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--n-trials", type=int, default=120)
    args = parser.parse_args(argv)

    path = build_pd_hpo_local_config(
        run_tag=args.run_tag,
        base_search_run_tag=args.base_search_run_tag,
        base_config_path=args.base_config_path,
        output_path=args.output_path,
        n_trials=args.n_trials,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
