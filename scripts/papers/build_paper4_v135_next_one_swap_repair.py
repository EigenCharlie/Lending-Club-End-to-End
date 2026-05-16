#!/usr/bin/env python3
"""Build Paper 4 v135 next one-swap repair artifacts."""

from __future__ import annotations

import json

from scripts.papers.paper4_one_swap_living_lab import build_repair_wave


def main() -> None:
    status = build_repair_wave(
        version=135,
        previous_repair_version=133,
        pricing_version=134,
        ordinal="twenty-seventh",
        next_reprice_version=136,
    )
    print(json.dumps({"v135": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
