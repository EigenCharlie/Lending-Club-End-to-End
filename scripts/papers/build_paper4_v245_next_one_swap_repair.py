#!/usr/bin/env python3
"""Build Paper 4 v245 next one-swap repair artifacts."""

from __future__ import annotations

import json

from scripts.papers.paper4_one_swap_living_lab import build_repair_wave


def main() -> None:
    status = build_repair_wave(
        version=245,
        previous_repair_version=243,
        pricing_version=244,
        ordinal="eighty-second",
        next_reprice_version=246,
    )
    print(json.dumps({"v245": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
