"""Export the canonical conformal method registry."""

from __future__ import annotations

import json
from pathlib import Path

from src.models.conformal_registry import build_conformal_method_registry


def main() -> None:
    out_path = Path("models/conformal_method_registry.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_conformal_method_registry()
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[conformal_method_registry] saved {out_path}")


if __name__ == "__main__":
    main()
