#!/usr/bin/env python3
"""Build Paper 4 v145 next one-swap repair artifacts."""

from __future__ import annotations

import json

from scripts.papers.paper4_one_swap_living_lab import build_repair_wave


def main() -> None:
    status = build_repair_wave(
        version=145,
        previous_repair_version=143,
        pricing_version=144,
        ordinal="thirty-second",
        next_reprice_version=146,
    )
    print(json.dumps({"v145": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
