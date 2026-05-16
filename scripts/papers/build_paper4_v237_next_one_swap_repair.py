#!/usr/bin/env python3
"""Build Paper 4 v237 next one-swap repair artifacts."""

from __future__ import annotations

import json

from scripts.papers.paper4_one_swap_living_lab import build_repair_wave


def main() -> None:
    status = build_repair_wave(
        version=237,
        previous_repair_version=235,
        pricing_version=236,
        ordinal="seventy-eighth",
        next_reprice_version=238,
    )
    print(json.dumps({"v237": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
