#!/usr/bin/env python3
"""Audit the CRPTO/IJDS champion-reopen environment without running searches."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = "paper1_crpto_reopen_ijds_2026_05_25"
CHAMPION_SELECTION = (
    ROOT
    / "models"
    / "portfolio_bound_aware"
    / "rank1_alpha01_bound_aware_276k_full_2026-04-05-1734"
    / "portfolio_bound_aware_selection.json"
)

KEY_TABLES = [
    "reports/paper_material/paper1/tables/paper1_bound_pareto_decision_summary_2026-05-25.csv",
    "reports/paper_material/paper1/tables/paper1_bound_pareto_nextwave_summary_2026-05-25.csv",
    "reports/paper_material/paper1/tables/paper1_return_aware_rerank_summary_2026-05-25.csv",
    "reports/paper_material/paper1/tables/paper1_conformal_reopen_candidate_gap_diagnostics_2026-05-25.csv",
    "reports/paper_material/paper1/tables/paper1_tableA2_robustness_frontier.csv",
    "reports/paper_material/paper1/tables/paper1_tableA3_nested_holdout.csv",
    "reports/paper_material/paper1/tables/paper1_table2_conformal_variant_benchmark.csv",
]

KEY_MEMOS = [
    "docs/research/paper1_champion_reopen_plan_2026-05-21.md",
    "docs/research/paper_estrella_journal_package_2026-05-04.md",
    "docs/research/repo_sync_closure_2026-05-04.md",
    "papers/paper_crpto_book/index.qmd",
    "papers/paper_crpto_book/chapters/01-ijds-target-and-claim.qmd",
    "papers/paper_crpto_book/chapters/02-book-to-crpto-intake.qmd",
    "papers/paper_crpto_book/chapters/03-manuscript-body.qmd",
    "papers/paper_crpto_book/chapters/06-open-boundaries.qmd",
]

SOURCE_DOCS = [
    {
        "topic": "CatBoost training parameters",
        "url": "https://catboost.ai/docs/en/references/training-parameters/common.html",
        "reopen_relevance": (
            "monotone constraints, feature weights, first-use penalties, "
            "Langevin/posterior sampling, and CPU/GPU support limits"
        ),
    },
    {
        "topic": "Optuna create_study / multi-objective",
        "url": "https://optuna.readthedocs.io/en/stable/reference/generated/optuna.create_study.html",
        "reopen_relevance": "persistent studies, directions, sampler/pruner governance",
    },
    {
        "topic": "MAPIE calibration metrics",
        "url": "https://mapie.readthedocs.io/en/latest/generated/mapie.metrics.calibration.expected_calibration_error.html",
        "reopen_relevance": "ECE and calibration diagnostics; risk-control APIs remain prototype-only",
    },
    {
        "topic": "NVIDIA cuOpt LP/QP features",
        "url": "https://docs.nvidia.com/cuopt/user-guide/latest/lp-qp-features.html",
        "reopen_relevance": "LP methods, batching, time limits, infeasibility, dual postsolve",
    },
    {
        "topic": "HiGHS Python interface",
        "url": "https://ergo-code.github.io/HiGHS/stable/interfaces/python/",
        "reopen_relevance": "direct highspy exact rerank and future warm-start/basis tests",
    },
    {
        "topic": "Pyomo APPSI",
        "url": "https://pyomo.readthedocs.io/en/stable/reference/topical/appsi/appsi.html",
        "reopen_relevance": "persistent solver interface for repeated LP experiments",
    },
]

PACKAGE_PROBE = r"""
import importlib
import importlib.metadata as md
import json
import sys

packages = {
    "catboost": ("catboost", "catboost"),
    "optuna": ("optuna", "optuna"),
    "mapie": ("mapie", "mapie"),
    "cuopt": ("cuopt", "cuopt"),
    "highspy": ("highspy", "highspy"),
    "pyomo": ("pyomo", "pyomo"),
    "scikit_learn": ("sklearn", "scikit-learn"),
    "pandas": ("pandas", "pandas"),
    "numpy": ("numpy", "numpy"),
    "cudf": ("cudf", "cudf"),
    "cuml": ("cuml", "cuml"),
    "cupy": ("cupy", "cupy"),
}

payload = {"python": sys.version.split()[0], "executable": sys.executable, "packages": {}}
for label, (module_name, dist_name) in packages.items():
    row = {}
    try:
        row["distribution_version"] = md.version(dist_name)
    except Exception as exc:
        row["distribution_error"] = str(exc)
    try:
        module = importlib.import_module(module_name)
        row["import_ok"] = True
        row["module_version"] = str(getattr(module, "__version__", row.get("distribution_version", "")))
    except Exception as exc:
        row["import_ok"] = False
        row["import_error"] = f"{type(exc).__name__}: {exc}"
    payload["packages"][label] = row

try:
    import mapie
    payload["mapie_capabilities"] = {
        "has_risk_control": bool(importlib.util.find_spec("mapie.risk_control")),
        "has_mondrian_module": bool(importlib.util.find_spec("mapie.mondrian")),
        "has_calibration_metrics": bool(importlib.util.find_spec("mapie.metrics.calibration")),
    }
except Exception as exc:
    payload["mapie_capabilities_error"] = str(exc)

print(json.dumps(payload, sort_keys=True))
"""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _run(cmd: list[str], *, timeout: int = 120) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except Exception as exc:
        return {"cmd": cmd, "returncode": None, "error": f"{type(exc).__name__}: {exc}"}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "path": str(path)}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _parse_git_status(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in text.splitlines():
        if not raw:
            continue
        if raw.startswith("## "):
            continue
        status = raw[:2]
        path = raw[3:].strip()
        bucket = "review"
        if path.startswith(
            ("data/processed/portfolio_bound_aware/", "models/portfolio_bound_aware/")
        ):
            bucket = "heavy_or_probe_do_not_commit_without_promotion"
        elif path.startswith("reports/run_logs/"):
            bucket = "ignored_runtime_log"
        elif path.startswith("reports/paper_material/paper1/tables/"):
            bucket = "semantic_table_review_for_promotion"
        elif path.startswith("scripts/search/"):
            bucket = "script_review_for_commit"
        rows.append({"status": status, "path": path, "bucket": bucket})
    return rows


def _file_summary(path: str) -> dict[str, Any]:
    full = ROOT / path
    row: dict[str, Any] = {"path": path, "exists": full.exists()}
    if full.exists():
        stat = full.stat()
        row.update(
            {
                "size_bytes": stat.st_size,
                "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
            }
        )
    return row


def _champion_snapshot() -> dict[str, Any]:
    payload = _load_json(CHAMPION_SELECTION)
    metrics = payload.get("selected_metrics", {}) if isinstance(payload, dict) else {}
    policy = payload.get("selected_policy", {}) if isinstance(payload, dict) else {}
    return {
        "path": str(CHAMPION_SELECTION.relative_to(ROOT)),
        "exists": CHAMPION_SELECTION.exists(),
        "return": metrics.get("realized_total_return"),
        "V": metrics.get("alpha01_weighted_miscoverage_V"),
        "Gamma_CP": metrics.get("alpha01_gamma_cp"),
        "violation": metrics.get("alpha01_violation"),
        "funded_coverage": metrics.get("alpha01_empirical_coverage_funded"),
        "alpha01_exact_pass": metrics.get("alpha01_exact_pass"),
        "policy": {
            "policy_mode": policy.get("policy_mode"),
            "risk_tolerance": policy.get("risk_tolerance"),
            "gamma": policy.get("gamma"),
            "uncertainty_aversion": policy.get("uncertainty_aversion"),
        },
    }


def _probe_python(python_executable: Path) -> dict[str, Any]:
    if not python_executable.exists():
        return {"executable": str(python_executable), "exists": False}
    result = _run([str(python_executable), "-c", PACKAGE_PROBE], timeout=180)
    row: dict[str, Any] = {"executable": str(python_executable), "exists": True, "raw": result}
    if result.get("returncode") == 0 and result.get("stdout"):
        try:
            row.update(json.loads(str(result["stdout"]).strip().splitlines()[-1]))
        except Exception as exc:
            row["parse_error"] = str(exc)
    return row


def _probe_conda_env(env_name: str) -> dict[str, Any]:
    probe = _run(["conda", "run", "-n", env_name, "python", "-c", PACKAGE_PROBE], timeout=240)
    conda_list = _run(["conda", "list", "-n", env_name], timeout=240)
    row: dict[str, Any] = {"env_name": env_name, "raw_probe": probe, "raw_conda_list": conda_list}
    if probe.get("returncode") == 0 and probe.get("stdout"):
        try:
            row.update(json.loads(str(probe["stdout"]).strip().splitlines()[-1]))
        except Exception as exc:
            row["probe_parse_error"] = str(exc)
    selected = []
    for line in str(conda_list.get("stdout", "")).splitlines():
        if re.match(
            r"^(cuopt|libcuopt|cuopt-cu|cudf|cudf-cu|cuml|cuml-cu|cupy|rmm|rmm-cu|cuvs|libcudf|libraft)\b",
            line,
        ):
            selected.append(line)
    row["selected_rapids_packages"] = selected
    joined = "\n".join(selected)
    row["mixed_rapids_stack_detected"] = bool(
        re.search(r"\b(cuopt|cudf|cuml|rmm|cuvs)\s+25\.12", joined)
        and re.search(r"\b(cuopt-cu|cudf-cu|libcuopt-cu|rmm-cu|libraft-cu).*\s26\.2", joined)
    )
    return row


def _write_dirty_manifest(path: Path, git_status: str, rows: list[dict[str, Any]]) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["bucket"]] = counts.get(row["bucket"], 0) + 1
    lines = [
        "# CRPTO/IJDS dirty manifest",
        f"generated_at_utc: {_utc_now()}",
        "",
        "## Bucket counts",
        *[f"- {key}: {value}" for key, value in sorted(counts.items())],
        "",
        "## Raw git status",
        "```",
        git_status.rstrip(),
        "```",
        "",
        "## Classified paths",
    ]
    for row in rows:
        lines.append(f"- {row['status']} {row['path']} :: {row['bucket']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_source_docs(path: Path) -> None:
    lines = [
        "# Source docs snapshot",
        f"generated_at_utc: {_utc_now()}",
        "",
        "These are the official documentation anchors used to govern the reopen plan.",
        "",
    ]
    for item in SOURCE_DOCS:
        lines.extend(
            [
                f"## {item['topic']}",
                f"- URL: {item['url']}",
                f"- Reopen relevance: {item['reopen_relevance']}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _recommendation(
    main_env: dict[str, Any],
    rapids_env: dict[str, Any],
    rapids_env_name: str,
) -> dict[str, Any]:
    main_packages = main_env.get("packages", {}) if isinstance(main_env, dict) else {}
    rapids_packages = rapids_env.get("packages", {}) if isinstance(rapids_env, dict) else {}
    main_ok = all(
        bool(main_packages.get(pkg, {}).get("import_ok"))
        for pkg in [
            "catboost",
            "optuna",
            "mapie",
            "highspy",
            "pyomo",
            "scikit_learn",
            "pandas",
            "numpy",
        ]
    )
    rapids_mixed = bool(rapids_env.get("mixed_rapids_stack_detected"))
    cuml_ok = bool(rapids_packages.get("cuml", {}).get("import_ok"))
    cuopt_ok = bool(rapids_packages.get("cuopt", {}).get("import_ok"))
    rapids_ok = bool(cuopt_ok and not rapids_mixed and cuml_ok)
    return {
        "main_venv_ok_for_pd_conformal_highs": main_ok,
        "rapids_ok_for_serious_cuopt": rapids_ok,
        "requires_clean_cuopt_env_before_serious_run": bool(
            rapids_mixed or not cuopt_ok or not cuml_ok
        ),
        "recommended_main_python": str(ROOT / ".venv" / "bin" / "python"),
        "recommended_cuopt_env": rapids_env_name if rapids_ok else "rapids-cuopt-2604-clean",
        "fallback_cuopt_env": "rapids-cuopt-2602-clean",
        "notes": [
            "Use .venv for CatBoost, Optuna, MAPIE, HiGHS, and Pyomo.",
            "Keep cuOpt in a separate clean RAPIDS environment; avoid installing PD packages there.",
            "Use a cuOpt ladder (Concurrent16->Concurrent8->PDLP16->Barrier4->Concurrent1) with HiGHS fallback.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default=os.environ.get("RUN_ROOT", DEFAULT_RUN_ROOT))
    parser.add_argument("--log-dir", default=str(ROOT / "reports" / "run_logs"))
    parser.add_argument("--rapids-env", default=os.environ.get("RAPIDS_ENV", "rapids"))
    args = parser.parse_args(argv)

    run_dir = Path(args.log_dir) / args.run_root
    run_dir.mkdir(parents=True, exist_ok=True)

    git_status_cmd = _run(["git", "status", "--short", "--branch", "--untracked-files=all"])
    git_status = str(git_status_cmd.get("stdout", ""))
    dirty_rows = _parse_git_status(git_status)

    main_env = _probe_python(ROOT / ".venv" / "bin" / "python")
    causal_env = _probe_python(ROOT / ".venv-causal" / "bin" / "python")
    rapids_env = _probe_conda_env(args.rapids_env)

    payload: dict[str, Any] = {
        "generated_at_utc": _utc_now(),
        "run_root": args.run_root,
        "repo_root": str(ROOT),
        "git": {
            "status": git_status_cmd,
            "branch": _run(["git", "branch", "--show-current"]),
            "head": _run(["git", "rev-parse", "HEAD"]),
            "main": _run(["git", "rev-parse", "main"]),
            "remote_head": _run(
                [
                    "git",
                    "rev-parse",
                    f"origin/{_run(['git', 'branch', '--show-current']).get('stdout', '').strip()}",
                ]
            ),
        },
        "dirty_classification": dirty_rows,
        "champion": _champion_snapshot(),
        "key_tables": [_file_summary(path) for path in KEY_TABLES],
        "key_memos": [_file_summary(path) for path in KEY_MEMOS],
        "environment": {
            "main_venv": main_env,
            "causal_venv": causal_env,
            "rapids": rapids_env,
            "conda_envs": _run(["conda", "env", "list"], timeout=120),
            "nvidia_smi": _run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                timeout=60,
            ),
        },
        "source_docs": SOURCE_DOCS,
    }
    payload["environment_recommendation"] = _recommendation(main_env, rapids_env, args.rapids_env)

    _write_json(run_dir / "env_audit.json", payload)
    _write_dirty_manifest(run_dir / "dirty_manifest.txt", git_status, dirty_rows)
    _write_source_docs(run_dir / "source_docs_snapshot.md")

    print(run_dir / "env_audit.json")
    print(json.dumps(payload["environment_recommendation"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
