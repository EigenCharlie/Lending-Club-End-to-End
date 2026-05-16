#!/usr/bin/env python3
"""Build Paper 4 v241 next one-swap repair artifacts."""

from __future__ import annotations

import json

from scripts.papers.paper4_one_swap_living_lab import build_repair_wave


def main() -> None:
    status = build_repair_wave(
        version=241,
        previous_repair_version=239,
        pricing_version=240,
        ordinal="eightieth",
        next_reprice_version=242,
    )
    print(json.dumps({"v241": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
