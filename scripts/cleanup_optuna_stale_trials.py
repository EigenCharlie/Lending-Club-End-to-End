#!/usr/bin/env python3
"""Mark stale Optuna RUNNING trials as FAIL without touching active trials.

Heuristic:
- trial state must be RUNNING
- last heartbeat (or start time if no heartbeat) older than threshold
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass
class RunningTrial:
    trial_id: int
    number: int
    study_name: str
    state: str
    datetime_start: str | None
    heartbeat: str | None


def _parse_dt(text: str | None) -> datetime | None:
    if not text:
        return None
    try:
        # SQLite rows often look like "2026-03-01 02:21:56.711298"
        return datetime.fromisoformat(str(text).replace(" ", "T")).replace(tzinfo=UTC)
    except Exception:
        return None


def _load_running_trials(con: sqlite3.Connection, study_name: str | None) -> list[RunningTrial]:
    query = """
    SELECT
      t.trial_id,
      t.number,
      s.study_name,
      CAST(t.state AS TEXT) AS state_text,
      t.datetime_start,
      MAX(h.heartbeat) AS last_heartbeat
    FROM trials t
    JOIN studies s ON s.study_id = t.study_id
    LEFT JOIN trial_heartbeats h ON h.trial_id = t.trial_id
    WHERE (CAST(t.state AS TEXT) = 'RUNNING' OR CAST(t.state AS TEXT) = '0')
    """
    params: list[str] = []
    if study_name:
        query += " AND s.study_name = ?"
        params.append(study_name)
    query += " GROUP BY t.trial_id, t.number, s.study_name, state_text, t.datetime_start"
    rows = con.execute(query, params).fetchall()
    return [
        RunningTrial(
            trial_id=int(r[0]),
            number=int(r[1]),
            study_name=str(r[2]),
            state=str(r[3]),
            datetime_start=str(r[4]) if r[4] is not None else None,
            heartbeat=str(r[5]) if r[5] is not None else None,
        )
        for r in rows
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Cleanup stale RUNNING Optuna trials.")
    parser.add_argument("--db-path", default="models/optuna_pd_catboost.db")
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--min-age-hours", type=float, default=6.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    db_path = Path(args.db_path).expanduser()
    if not db_path.exists():
        raise FileNotFoundError(f"Optuna DB not found: {db_path}")

    min_age_seconds = max(300.0, float(args.min_age_hours) * 3600.0)
    now = datetime.now(UTC)

    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        trials = _load_running_trials(con, args.study_name)
        stale: list[RunningTrial] = []
        for tr in trials:
            last_seen = _parse_dt(tr.heartbeat) or _parse_dt(tr.datetime_start)
            if last_seen is None:
                continue
            age_seconds = (now - last_seen).total_seconds()
            if age_seconds >= min_age_seconds:
                stale.append(tr)

        report = {
            "generated_at_utc": now.isoformat(),
            "db_path": str(db_path),
            "study_name_filter": args.study_name,
            "min_age_hours": float(args.min_age_hours),
            "running_trials_found": int(len(trials)),
            "stale_trials_found": int(len(stale)),
            "stale_trials": [
                {
                    "trial_id": t.trial_id,
                    "number": t.number,
                    "study_name": t.study_name,
                    "datetime_start": t.datetime_start,
                    "heartbeat": t.heartbeat,
                }
                for t in stale
            ],
            "dry_run": bool(args.dry_run),
        }

        if not stale or args.dry_run:
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 0

        ts = now.strftime("%Y%m%d_%H%M%S")
        backup_path = db_path.with_suffix(db_path.suffix + f".bak-stale-clean-{ts}")
        shutil.copy2(db_path, backup_path)

        now_sql = now.replace(tzinfo=None).isoformat(sep=" ")
        for tr in stale:
            con.execute(
                "UPDATE trials SET state = ?, datetime_complete = ? WHERE trial_id = ?",
                ("FAIL", now_sql, tr.trial_id),
            )
        con.commit()
        report["backup_path"] = str(backup_path)
        report["updated_trial_ids"] = [t.trial_id for t in stale]
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
