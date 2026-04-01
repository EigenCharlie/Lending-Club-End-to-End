#!/usr/bin/env bash

echo "HISTORICAL wrapper: delegating to scripts/history/run_paper_grade_pre_quarto.sh" >&2
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/history/run_paper_grade_pre_quarto.sh" "$@"
