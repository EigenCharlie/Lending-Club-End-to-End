#!/usr/bin/env python3
"""Build Paper 4 v187 next one-swap repair artifacts."""

from __future__ import annotations

import json

from scripts.papers.paper4_one_swap_living_lab import build_repair_wave


def main() -> None:
    status = build_repair_wave(
        version=187,
        previous_repair_version=185,
        pricing_version=186,
        ordinal="fifty-third",
        next_reprice_version=188,
    )
    print(json.dumps({"v187": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
