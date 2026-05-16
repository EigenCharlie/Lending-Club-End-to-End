#!/usr/bin/env python3
"""Build Paper 4 v160 post-v159 one-swap repricing artifacts."""

from __future__ import annotations

import json

from scripts.papers.paper4_one_swap_living_lab import build_reprice_wave


def main() -> None:
    status = build_reprice_wave(version=160, previous_repair_version=159, next_repair_version=161)
    print(json.dumps({"v160": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
