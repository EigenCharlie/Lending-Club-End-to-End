#!/usr/bin/env python3
"""Build Paper 4 v159 next one-swap repair artifacts."""

from __future__ import annotations

import json

from scripts.papers.paper4_one_swap_living_lab import build_repair_wave


def main() -> None:
    status = build_repair_wave(
        version=159,
        previous_repair_version=157,
        pricing_version=158,
        ordinal="thirty-ninth",
        next_reprice_version=160,
    )
    print(json.dumps({"v159": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
