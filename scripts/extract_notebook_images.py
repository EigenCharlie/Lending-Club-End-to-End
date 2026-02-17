"""Extract embedded PNG outputs from notebooks into reproducible image assets.

Usage:
    uv run python scripts/extract_notebook_images.py
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import nbformat

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "notebook_images"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"


def _normalize_markdown(md: str) -> str:
    lines = [line.strip() for line in md.splitlines() if line.strip()]
    if not lines:
        return ""
    # Prefer heading-like lines; fallback to first non-empty line.
    for line in lines:
        if line.startswith("#") or line.startswith("---"):
            return line.lstrip("-# ").strip()
    return lines[0]


def _nearest_markdown(cells: list, code_cell_index: int) -> str:
    for idx in range(code_cell_index - 1, -1, -1):
        cell = cells[idx]
        if cell.get("cell_type") == "markdown":
            text = _normalize_markdown(cell.get("source", ""))
            if text:
                return text
    return ""


def _decode_png(payload: str | list[str]) -> bytes:
    if isinstance(payload, list):
        payload = "".join(payload)
    return base64.b64decode(payload)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    total_images = 0

    notebook_paths = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    for nb_path in notebook_paths:
        nb = nbformat.read(nb_path, as_version=4)
        notebook_name = nb_path.stem
        notebook_out_dir = OUTPUT_DIR / notebook_name
        notebook_out_dir.mkdir(parents=True, exist_ok=True)

        notebook_image_count = 0
        for cell_idx, cell in enumerate(nb.cells):
            if cell.get("cell_type") != "code":
                continue

            context = _nearest_markdown(nb.cells, cell_idx)
            outputs = cell.get("outputs", [])
            for out_idx, output in enumerate(outputs):
                data = output.get("data", {})
                if not isinstance(data, dict) or "image/png" not in data:
                    continue

                image_name = f"cell_{cell_idx:03d}_out_{out_idx:02d}.png"
                image_rel_path = Path("reports") / "notebook_images" / notebook_name / image_name
                image_abs_path = PROJECT_ROOT / image_rel_path
                image_abs_path.write_bytes(_decode_png(data["image/png"]))

                text_output = ""
                if "text/plain" in data:
                    text_payload = data["text/plain"]
                    if isinstance(text_payload, list):
                        text_output = "\n".join(text_payload).strip()
                    else:
                        text_output = str(text_payload).strip()

                manifest.append(
                    {
                        "notebook": nb_path.name,
                        "notebook_stem": notebook_name,
                        "cell_index": cell_idx,
                        "output_index": out_idx,
                        "image_path": str(image_rel_path),
                        "context": context,
                        "text_output": text_output[:300],
                    }
                )
                notebook_image_count += 1
                total_images += 1

        print(f"{nb_path.name}: extracted {notebook_image_count} images")

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved manifest: {MANIFEST_PATH}")
    print(f"Total images extracted: {total_images}")


if __name__ == "__main__":
    main()
