#!/usr/bin/env python3
"""Build Paper 4 v197 next one-swap repair artifacts."""

from __future__ import annotations

import json

from scripts.papers.paper4_one_swap_living_lab import build_repair_wave


def main() -> None:
    status = build_repair_wave(
        version=197,
        previous_repair_version=195,
        pricing_version=196,
        ordinal="fifty-eighth",
        next_reprice_version=198,
    )
    print(json.dumps({"v197": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
