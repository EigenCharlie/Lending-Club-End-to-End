#!/usr/bin/env python3
"""Build Paper 4 v194 post-v193 one-swap repricing artifacts."""

from __future__ import annotations

import json

from scripts.papers.paper4_one_swap_living_lab import build_reprice_wave


def main() -> None:
    status = build_reprice_wave(version=194, previous_repair_version=193, next_repair_version=195)
    print(json.dumps({"v194": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
