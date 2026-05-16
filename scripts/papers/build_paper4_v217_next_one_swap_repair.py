#!/usr/bin/env python3
"""Build Paper 4 v217 next one-swap repair artifacts."""

from __future__ import annotations

import json

from scripts.papers.paper4_one_swap_living_lab import build_repair_wave


def main() -> None:
    status = build_repair_wave(
        version=217,
        previous_repair_version=215,
        pricing_version=216,
        ordinal="sixty-eighth",
        next_reprice_version=218,
    )
    print(json.dumps({"v217": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
