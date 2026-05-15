#!/usr/bin/env python3
"""Build Paper 4 v115 next one-swap repair artifacts."""

from __future__ import annotations

import json

from scripts.papers.paper4_one_swap_living_lab import build_repair_wave


def main() -> None:
    status = build_repair_wave(
        version=115,
        previous_repair_version=113,
        pricing_version=114,
        ordinal="seventeenth",
        next_reprice_version=116,
    )
    print(json.dumps({"v115": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
