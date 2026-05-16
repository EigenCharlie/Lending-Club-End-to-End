#!/usr/bin/env python3
"""Build Paper 4 v123 next one-swap repair artifacts."""

from __future__ import annotations

import json

from scripts.papers.paper4_one_swap_living_lab import build_repair_wave


def main() -> None:
    status = build_repair_wave(
        version=123,
        previous_repair_version=121,
        pricing_version=122,
        ordinal="twenty-first",
        next_reprice_version=124,
    )
    print(json.dumps({"v123": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
