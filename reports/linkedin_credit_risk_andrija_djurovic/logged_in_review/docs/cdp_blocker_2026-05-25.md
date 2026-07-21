# Andrija Logged-In P0/P1 CDP Resolution - 2026-05-25

## Status

The original Chrome CDP attempt was blocked, but the logged-in P0/P1 pass was
completed after switching to the Windows Opera GX profile used in the earlier
LinkedIn workflow.

- Queue: `reports/linkedin_credit_risk_andrija_djurovic/logged_in_review/data/logged_in_review_queue.csv`
- Rows: 37
- P0 rows: 12
- P1 rows: 25
- Final capture rows: 37
- Completed rendered captures: 34
- Blocked/checkpoint rows: 2
- Capture error rows: 1
- Visible comments captured: 121
- External link rows captured: 72

The initial capture attempt failed because no user-owned Chrome DevTools
endpoint was available at `http://127.0.0.1:9222`.

Command attempted:

```bash
python scripts/research/capture_linkedin_logged_in_cdp.py \
  --pack-dir reports/linkedin_credit_risk_andrija_djurovic/logged_in_review \
  --items all \
  --sleep-seconds 1.0 \
  --expand-iterations 6
```

Observed blocker:

```text
CDP endpoint unavailable at http://127.0.0.1:9222. Launch a user-owned Chrome session with --remote-debugging-port=9222 first.
```

## Resume Command

The working path was to run Playwright from Windows against a visible Opera GX
session launched with remote debugging and a non-destructive copy of the user's
Opera profile:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File scripts\research\run_andrija_logged_in_p0_p1_windows.ps1 `
  -Items all `
  -Port 9224 `
  -ChromeProfile "$env:TEMP\codex-linkedin-opera-profile-copy" `
  -BrowserPath "$env:LOCALAPPDATA\Programs\Opera GX\opera.exe" `
  -ProfileDirectory "" `
  -ExpandIterations 6 `
  -SleepSeconds 1.0
```

If a future pass uses Chrome instead, the same script can be reused with a
Chrome executable and profile path.

## Windows Browser Launch Pattern

Use a separate temporary browser profile copy so normal browsing is not
disturbed:

```powershell
& "$env:LOCALAPPDATA\Programs\Opera GX\opera.exe" `
  --remote-debugging-port=9222 `
  --user-data-dir="$env:TEMP\codex-linkedin-cdp"
```

Then sign in to LinkedIn in that visible browser window and keep it open while
the capture runs.

## Guardrails

- Use only the user's own visible access.
- Do not paste passwords or cookie values into chat.
- Do not bypass captcha/checkpoints, use fake accounts, or evade rate limits.
- LinkedIn comments remain intake/source-discovery only unless backed by an
  independent source or local project result.
