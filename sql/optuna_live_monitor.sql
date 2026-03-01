-- Live monitor for Optuna study: pd_catboost_optuna_temporal
-- Use with SQLTools -> connection: "Optuna PD SQLite"

-- 1) Global status
SELECT
  COUNT(*) AS total_trials,
  SUM(CASE WHEN state = 'COMPLETE' THEN 1 ELSE 0 END) AS complete_trials,
  SUM(CASE WHEN state = 'PRUNED' THEN 1 ELSE 0 END) AS pruned_trials,
  SUM(CASE WHEN state = 'RUNNING' THEN 1 ELSE 0 END) AS running_trials,
  MAX(number) AS max_trial_number
FROM trials;

-- 2) Best AUC (validation) so far
SELECT
  t.number AS trial_number,
  tv.value AS best_auc,
  t.datetime_start,
  t.datetime_complete
FROM trials t
JOIN trial_values tv ON tv.trial_id = t.trial_id
WHERE t.state = 'COMPLETE'
ORDER BY tv.value DESC
LIMIT 1;

-- 3) Running jobs (active trial(s))
SELECT
  number AS running_trial,
  datetime_start,
  ROUND((julianday('now') - julianday(datetime_start)) * 24 * 60, 1) AS running_minutes
FROM trials
WHERE state = 'RUNNING'
ORDER BY number;

-- 4) Latest finished trials with AUC and duration
SELECT
  t.number AS trial_number,
  tv.value AS auc_validation,
  t.state,
  t.datetime_start,
  t.datetime_complete,
  ROUND((julianday(t.datetime_complete) - julianday(t.datetime_start)) * 24 * 60, 2) AS duration_min
FROM trials t
LEFT JOIN trial_values tv ON tv.trial_id = t.trial_id
WHERE t.datetime_complete IS NOT NULL
ORDER BY t.number DESC
LIMIT 25;

-- 5) Best 20 trials (leaderboard)
SELECT
  t.number AS trial_number,
  tv.value AS auc_validation,
  t.datetime_complete
FROM trials t
JOIN trial_values tv ON tv.trial_id = t.trial_id
WHERE t.state = 'COMPLETE'
ORDER BY tv.value DESC
LIMIT 20;

-- 6) Trials remaining (set your target total)
-- For this project, if baseline had 800 and run adds 400, target_total_trials=1200.
WITH cfg AS (
  SELECT 1200 AS target_total_trials
),
cur AS (
  SELECT COALESCE(MAX(number), -1) + 1 AS observed_total_trials
  FROM trials
)
SELECT
  cfg.target_total_trials,
  cur.observed_total_trials,
  MAX(0, cfg.target_total_trials - cur.observed_total_trials) AS remaining_trials
FROM cfg, cur;

-- 7) Newest completed trial + best-so-far delta
WITH best AS (
  SELECT MAX(tv.value) AS best_auc
  FROM trials t
  JOIN trial_values tv ON tv.trial_id = t.trial_id
  WHERE t.state = 'COMPLETE'
),
last_completed AS (
  SELECT t.number, tv.value AS auc_validation, t.datetime_complete
  FROM trials t
  JOIN trial_values tv ON tv.trial_id = t.trial_id
  WHERE t.state = 'COMPLETE'
  ORDER BY t.number DESC
  LIMIT 1
)
SELECT
  l.number AS last_trial,
  l.auc_validation AS last_auc,
  b.best_auc,
  (b.best_auc - l.auc_validation) AS gap_to_best_auc,
  l.datetime_complete
FROM last_completed l, best b;

-- 8) Stale RUNNING trials (likely orphaned from crashes/restarts)
SELECT
  number AS running_trial,
  datetime_start,
  ROUND((julianday('now') - julianday(datetime_start)) * 24 * 60, 1) AS running_minutes
FROM trials
WHERE state = 'RUNNING'
  AND (julianday('now') - julianday(datetime_start)) * 24 * 60 > 30
ORDER BY number;

-- 9) One-row dashboard (trial actual, AUC best, AUC último, faltantes, estado)
-- Adjust target_total_trials if your run plan changes.
WITH cfg AS (
  SELECT 1200 AS target_total_trials
),
counts AS (
  SELECT
    COALESCE(MAX(number), -1) + 1 AS observed_total_trials,
    SUM(CASE WHEN state = 'RUNNING' THEN 1 ELSE 0 END) AS running_trials
  FROM trials
),
current_trial AS (
  SELECT MAX(number) AS trial_actual
  FROM trials
  WHERE state = 'RUNNING'
),
best_trial AS (
  SELECT
    t.number AS best_trial,
    tv.value AS auc_best
  FROM trials t
  JOIN trial_values tv ON tv.trial_id = t.trial_id
  WHERE t.state = 'COMPLETE'
  ORDER BY tv.value DESC
  LIMIT 1
),
last_completed AS (
  SELECT
    t.number AS last_trial,
    tv.value AS auc_last
  FROM trials t
  JOIN trial_values tv ON tv.trial_id = t.trial_id
  WHERE t.state = 'COMPLETE'
  ORDER BY t.number DESC
  LIMIT 1
)
SELECT
  CASE
    WHEN counts.running_trials > 0 THEN 'RUNNING'
    ELSE 'IDLE'
  END AS estado_run,
  current_trial.trial_actual,
  last_completed.last_trial,
  ROUND(last_completed.auc_last, 9) AS auc_last,
  best_trial.best_trial,
  ROUND(best_trial.auc_best, 9) AS auc_best,
  cfg.target_total_trials,
  counts.observed_total_trials,
  MAX(0, cfg.target_total_trials - counts.observed_total_trials) AS remaining_trials
FROM cfg
CROSS JOIN counts
LEFT JOIN current_trial ON 1 = 1
LEFT JOIN last_completed ON 1 = 1
LEFT JOIN best_trial ON 1 = 1;
