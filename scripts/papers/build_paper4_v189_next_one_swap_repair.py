#!/usr/bin/env python3
"""Build Paper 4 v189 next one-swap repair artifacts."""

from __future__ import annotations

import json

from scripts.papers.paper4_one_swap_living_lab import build_repair_wave


def main() -> None:
    status = build_repair_wave(
        version=189,
        previous_repair_version=187,
        pricing_version=188,
        ordinal="fifty-fourth",
        next_reprice_version=190,
    )
    print(json.dumps({"v189": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
