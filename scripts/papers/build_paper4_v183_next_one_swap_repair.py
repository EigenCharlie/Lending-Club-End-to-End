#!/usr/bin/env python3
"""Build Paper 4 v183 next one-swap repair artifacts."""

from __future__ import annotations

import json

from scripts.papers.paper4_one_swap_living_lab import build_repair_wave


def main() -> None:
    status = build_repair_wave(
        version=183,
        previous_repair_version=181,
        pricing_version=182,
        ordinal="fifty-first",
        next_reprice_version=184,
    )
    print(json.dumps({"v183": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
