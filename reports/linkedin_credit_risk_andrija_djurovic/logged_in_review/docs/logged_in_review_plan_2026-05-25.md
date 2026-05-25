# Andrija Logged-In Review Queue - 2026-05-25

This queue contains P0 and P1 targets from
`reports/linkedin_credit_risk_andrija_djurovic/data/andrija_login_only_priority_queue.csv`.

- Queue rows: 37
- Intended method: Playwright/CDP against a user-owned visible logged-in browser
  session.
- Scope: rendered post text, visible comments, comment links, and classification
  of profile activity IDs that did not expose public own-post permalinks.

## Guardrails

- No fake accounts, captcha bypass, stealth, or rate evasion.
- Do not print cookie values or credentials.
- Treat comments and LinkedIn-only materials as private research intake.
- Promote no paper/book claims from comments alone.

## Stop Condition

Close each row when logged-in rendered text, visible comments, comment links,
and newly exposed attachments are captured/read, or when the page is classified
as non-credit-risk, reaction-only, inaccessible, or already covered by stronger
external sources.
