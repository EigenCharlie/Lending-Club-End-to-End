#!/usr/bin/env python3
"""Build Paper 4 v163 next one-swap repair artifacts."""

from __future__ import annotations

import json

from scripts.papers.paper4_one_swap_living_lab import build_repair_wave


def main() -> None:
    status = build_repair_wave(
        version=163,
        previous_repair_version=161,
        pricing_version=162,
        ordinal="forty-first",
        next_reprice_version=164,
    )
    print(json.dumps({"v163": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
