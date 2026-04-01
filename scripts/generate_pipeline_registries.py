"""Render tracked pipeline registries as JSON snapshots."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_DIR = ROOT / "configs" / "pipeline_registry"
OUT_DIR = ROOT / "models" / "pipeline_registry"


def _load_yaml(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return dict(payload) if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    for yaml_path in sorted(REGISTRY_DIR.glob("*.yaml")):
        payload = _load_yaml(yaml_path)
        out_path = OUT_DIR / f"{yaml_path.stem}.json"
        _write_json(out_path, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
