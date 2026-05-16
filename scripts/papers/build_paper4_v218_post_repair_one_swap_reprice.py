#!/usr/bin/env python3
"""Build Paper 4 v218 post-v217 one-swap repricing artifacts."""

from __future__ import annotations

import json

from scripts.papers.paper4_one_swap_living_lab import build_reprice_wave


def main() -> None:
    status = build_reprice_wave(version=218, previous_repair_version=217, next_repair_version=219)
    print(json.dumps({"v218": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
