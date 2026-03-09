#!/usr/bin/env python3
"""Wait for minimum memory/swap headroom before starting heavy jobs.

This reduces OOM risk on WSL where concurrent services (Streamlit, notebooks,
GPU benches) can quickly exhaust RAM+swap.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

MEMINFO_PATH = Path("/proc/meminfo")


@dataclass
class MemSnapshot:
    mem_available_gb: float
    swap_free_gb: float
    total_headroom_gb: float


def _read_meminfo_bytes() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in MEMINFO_PATH.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        token = rest.strip().split()[0]
        values[key.strip()] = int(token) * 1024
    return values


def _snapshot() -> MemSnapshot:
    info = _read_meminfo_bytes()
    mem_available = float(info.get("MemAvailable", 0))
    swap_free = float(info.get("SwapFree", 0))
    gb = float(1024**3)
    return MemSnapshot(
        mem_available_gb=mem_available / gb,
        swap_free_gb=swap_free / gb,
        total_headroom_gb=(mem_available + swap_free) / gb,
    )


def _ok(
    snap: MemSnapshot,
    *,
    min_mem_gb: float,
    min_swap_gb: float,
    min_total_headroom_gb: float,
) -> bool:
    return (
        snap.mem_available_gb >= min_mem_gb
        and snap.swap_free_gb >= min_swap_gb
        and snap.total_headroom_gb >= min_total_headroom_gb
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wait for memory headroom before heavy steps.")
    p.add_argument("--label", default="step", help="Short label for logs")
    p.add_argument("--min-mem-gb", type=float, default=6.0, help="Minimum MemAvailable GB")
    p.add_argument("--min-swap-gb", type=float, default=3.0, help="Minimum SwapFree GB")
    p.add_argument(
        "--min-total-headroom-gb",
        type=float,
        default=10.0,
        help="Minimum MemAvailable+SwapFree GB",
    )
    p.add_argument(
        "--max-wait-seconds",
        type=int,
        default=1800,
        help="Abort after waiting this long",
    )
    p.add_argument(
        "--poll-seconds",
        type=int,
        default=20,
        help="Polling interval while waiting",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    start = time.monotonic()
    while True:
        snap = _snapshot()
        ok = _ok(
            snap,
            min_mem_gb=float(args.min_mem_gb),
            min_swap_gb=float(args.min_swap_gb),
            min_total_headroom_gb=float(args.min_total_headroom_gb),
        )
        print(
            "[mem-guard] "
            f"label={args.label} "
            f"mem_available_gb={snap.mem_available_gb:.2f} "
            f"swap_free_gb={snap.swap_free_gb:.2f} "
            f"total_headroom_gb={snap.total_headroom_gb:.2f} "
            f"target=({args.min_mem_gb:.1f},{args.min_swap_gb:.1f},{args.min_total_headroom_gb:.1f}) "
            f"ok={int(ok)}",
            flush=True,
        )
        if ok:
            return 0
        elapsed = time.monotonic() - start
        if elapsed >= int(args.max_wait_seconds):
            print(
                f"[mem-guard] label={args.label} timeout_s={int(elapsed)} status=failed",
                flush=True,
            )
            return 1
        time.sleep(max(1, int(args.poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
