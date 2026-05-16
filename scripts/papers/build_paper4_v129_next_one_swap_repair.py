#!/usr/bin/env python3
"""Build Paper 4 v129 next one-swap repair artifacts."""

from __future__ import annotations

import json

from scripts.papers.paper4_one_swap_living_lab import build_repair_wave


def main() -> None:
    status = build_repair_wave(
        version=129,
        previous_repair_version=127,
        pricing_version=128,
        ordinal="twenty-fourth",
        next_reprice_version=130,
    )
    print(json.dumps({"v129": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
