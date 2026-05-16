#!/usr/bin/env python3
"""Build Paper 4 v225 next one-swap repair artifacts."""

from __future__ import annotations

import json

from scripts.papers.paper4_one_swap_living_lab import build_repair_wave


def main() -> None:
    status = build_repair_wave(
        version=225,
        previous_repair_version=223,
        pricing_version=224,
        ordinal="seventy-second",
        next_reprice_version=226,
    )
    print(json.dumps({"v225": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
