#!/usr/bin/env python3
"""Build Paper 4 v106 post-v105 one-swap repricing artifacts."""

from __future__ import annotations

import json

from scripts.papers.paper4_one_swap_living_lab import build_reprice_wave


def main() -> None:
    status = build_reprice_wave(version=106, previous_repair_version=105, next_repair_version=107)
    print(json.dumps({"v106": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
