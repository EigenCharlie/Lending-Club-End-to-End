"""Regenerate reports/dependency_summary.json from repo config, lockfile and code usage."""

from __future__ import annotations

import argparse
import json
import subprocess
import tomllib
from pathlib import Path
from typing import Any

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from src.utils.pipeline_runtime import (
    atomic_write_text,
    write_last_valid_artifact,
    write_runtime_checkpoint,
    write_runtime_status,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "reports" / "dependency_summary.json"

PACKAGE_IMPORT_MAP: dict[str, list[str]] = {
    "betacal": ["betacal", "BetaCalibration"],
    "catboost": ["catboost"],
    "CausalPy": ["causalpy", "CausalPy"],
    "crepes": ["crepes"],
    "dagshub": ["dagshub"],
    "dbt-duckdb": ["dbt"],
    "dowhy": ["dowhy"],
    "duckdb": ["duckdb"],
    "dvc": ["dvc"],
    "econml": ["econml"],
    "fairlearn": ["fairlearn"],
    "fastapi": ["fastapi"],
    "feast": ["feast"],
    "hierarchicalforecast": ["hierarchicalforecast"],
    "highspy": ["highspy"],
    "httpx": ["httpx"],
    "ipykernel": ["ipykernel"],
    "joblib": ["joblib"],
    "jupyter": ["jupyter"],
    "lightgbm": ["lightgbm"],
    "lifelines": ["lifelines"],
    "mapie": ["mapie"],
    "mlflow": ["mlflow"],
    "optbinning": ["optbinning"],
    "optuna": ["optuna"],
    "optuna-integration": ["optuna.integration", "optuna_integration"],
    "pandera": ["pandera"],
    "polars": ["polars"],
    "pyepo": ["pyepo"],
    "pyomo": ["pyomo"],
    "scikit-survival": ["sksurv"],
    "skops": ["skops"],
    "streamlit": ["streamlit"],
    "venn-abers": ["venn_abers"],
}

PACKAGE_NOTES: dict[str, str] = {
    "betacal": "Beta calibration backend used as the fourth calibration candidate in PD evaluation.",
    "crepes": "Research-only benchmark lane for predictive systems/p-values; not official conformal stack.",
    "lightgbm": "Keep as PD challenger and cheap LGD/EAD baseline, not as champion replacement by default.",
    "feast": "Platform-only extra; intentionally excluded from the canonical rerun surface.",
    "polars": "Useful for side benchmarks/tooling, not part of the main canonical rerun.",
    "CausalPy": "Separate causal environment only; keep out of the official causal stack.",
}
MANUAL_PACKAGES: dict[str, dict[str, Any]] = {
    "betacal": {
        "name": "betacal",
        "repo_version": None,
        "sources": ["transitive (via mapie)"],
    }
}


def _normalize_package_name(name: str) -> str:
    return name.lower().replace("_", "-")


def _load_pyproject_requirements() -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    payload = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = payload.get("project", {})
    deps: dict[str, dict[str, Any]] = {}
    source_lookup: dict[str, str] = {}

    for raw in project.get("dependencies", []) or []:
        req = Requirement(str(raw))
        key = _normalize_package_name(req.name)
        deps[key] = {
            "name": req.name,
            "repo_version": str(req.specifier) or None,
            "sources": ["pyproject"],
        }
        source_lookup[key] = "pyproject"

    for group, raw_deps in (project.get("optional-dependencies", {}) or {}).items():
        for raw in raw_deps or []:
            req = Requirement(str(raw))
            key = _normalize_package_name(req.name)
            source = f"optional:{group}"
            if key not in deps:
                deps[key] = {
                    "name": req.name,
                    "repo_version": str(req.specifier) or None,
                    "sources": [source],
                }
            else:
                deps[key]["sources"] = sorted({*deps[key]["sources"], source})
                if not deps[key].get("repo_version"):
                    deps[key]["repo_version"] = str(req.specifier) or None
            source_lookup[key] = source
    return deps, source_lookup


def _load_requirement_files(existing: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = dict(existing)
    for req_file in sorted((REPO_ROOT / "requirements").glob("*.txt")):
        for raw_line in req_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            req = Requirement(line)
            key = _normalize_package_name(req.name)
            source = str(req_file.relative_to(REPO_ROOT))
            if key not in out:
                out[key] = {
                    "name": req.name,
                    "repo_version": str(req.specifier) or None,
                    "sources": [source],
                }
            else:
                out[key]["sources"] = sorted({*out[key]["sources"], source})
                if not out[key].get("repo_version"):
                    out[key]["repo_version"] = str(req.specifier) or None
    return out


def _load_lock_versions() -> dict[str, str]:
    payload = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for pkg in payload.get("package", []) or []:
        name = _normalize_package_name(str(pkg.get("name", "")))
        version = str(pkg.get("version", "")).strip()
        if name and version and name not in out:
            out[name] = version
    return out


def _is_up_to_date(repo_version: str | None, latest: str | None) -> bool | None:
    if not repo_version or not latest:
        return None
    spec = repo_version.strip()
    if any(token in spec for token in "<>!=~"):
        try:
            return Version(latest) in SpecifierSet(spec)
        except Exception:
            return None
    return spec == latest


def _find_usage_files(package_name: str) -> list[str]:
    patterns = PACKAGE_IMPORT_MAP.get(package_name, [package_name])
    files: set[str] = set()
    for pattern in patterns:
        try:
            result = subprocess.run(
                [
                    "rg",
                    "-l",
                    pattern,
                    "src",
                    "scripts",
                    "tests",
                    "api",
                    "streamlit_app",
                    "feature_repo",
                    "docs",
                    "book",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            return []
        for line in result.stdout.splitlines():
            cleaned = line.strip()
            if cleaned:
                files.add(cleaned)
    return sorted(files)


def _build_summary_rows() -> list[dict[str, Any]]:
    base, _ = _load_pyproject_requirements()
    deps = _load_requirement_files(base)
    for key, payload in MANUAL_PACKAGES.items():
        deps.setdefault(key, dict(payload))
    lock_versions = _load_lock_versions()
    rows: list[dict[str, Any]] = []
    for key in sorted(deps):
        row = dict(deps[key])
        latest = lock_versions.get(key)
        repo_version = row.get("repo_version")
        usage_files = _find_usage_files(str(row["name"]))
        rows.append(
            {
                "name": row["name"],
                "repo_version": repo_version,
                "latest": latest or repo_version,
                "up_to_date": _is_up_to_date(repo_version, latest or repo_version),
                "sources": row.get("sources", []),
                "files": usage_files,
                "file_count": len(usage_files),
                "notes": PACKAGE_NOTES.get(str(row["name"]), ""),
            }
        )
    return rows


def main(output_path: str = str(OUTPUT_PATH)) -> None:
    stage_name = "dependency_audit"
    write_runtime_status(stage_name, phase="scanning_repo", state="running")
    rows = _build_summary_rows()
    write_runtime_checkpoint(
        stage_name,
        "dependency_rows_built",
        {
            "dependency_count": int(len(rows)),
            "dependencies_with_usage": int(
                sum(1 for row in rows if int(row.get("file_count", 0)) > 0)
            ),
        },
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(target, json.dumps(rows, indent=2))
    write_last_valid_artifact(
        stage_name,
        artifact_key="dependency_summary",
        artifact_path=target,
        extra={"dependency_count": int(len(rows))},
    )
    write_runtime_status(
        stage_name,
        phase="completed",
        state="completed",
        extra={"output_path": str(target), "dependency_count": int(len(rows))},
    )
    print(f"Wrote {len(rows)} dependency rows to {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()
    main(output_path=args.output)
