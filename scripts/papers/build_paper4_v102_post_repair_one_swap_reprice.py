#!/usr/bin/env python3
"""Build Paper 4 v102 post-v101 one-swap repricing artifacts."""

from __future__ import annotations

import json

from scripts.papers.paper4_one_swap_living_lab import build_reprice_wave


def main() -> None:
    status = build_reprice_wave(version=102, previous_repair_version=101, next_repair_version=103)
    print(json.dumps({"v102": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
