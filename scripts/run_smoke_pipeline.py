"""Canonical smoke/core pipeline entrypoint."""

from __future__ import annotations

import argparse

from scripts.end_to_end_pipeline import main as _main


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="smoke")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--skip-make-dataset", action="store_true")
    args = parser.parse_args()
    return _main(
        run_name=args.run_name,
        continue_on_error=args.continue_on_error,
        skip_make_dataset=args.skip_make_dataset,
    )


if __name__ == "__main__":
    raise SystemExit(main())
