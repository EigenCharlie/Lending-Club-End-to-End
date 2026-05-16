#!/usr/bin/env python3
"""Build Paper 4 v203 next one-swap repair artifacts."""

from __future__ import annotations

import json

from scripts.papers.paper4_one_swap_living_lab import build_repair_wave


def main() -> None:
    status = build_repair_wave(
        version=203,
        previous_repair_version=201,
        pricing_version=202,
        ordinal="sixty-first",
        next_reprice_version=204,
    )
    print(json.dumps({"v203": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
