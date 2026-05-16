#!/usr/bin/env python3
"""Build Paper 4 v253 next one-swap repair artifacts."""

from __future__ import annotations

import json

from scripts.papers.paper4_one_swap_living_lab import build_repair_wave


def main() -> None:
    status = build_repair_wave(
        version=253,
        previous_repair_version=251,
        pricing_version=252,
        ordinal="post-v251",
        next_reprice_version=254,
    )
    print(json.dumps({"v253": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
