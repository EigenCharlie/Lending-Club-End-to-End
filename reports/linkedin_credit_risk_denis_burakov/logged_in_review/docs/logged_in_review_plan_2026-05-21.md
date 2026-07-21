# Logged-In LinkedIn Review Queue - 2026-05-21

This pack is for a third review pass using the user's own visible logged-in
LinkedIn access. It is intentionally separate from the public first and second
ingests.

- Queue rows: 80
- Capture method: Chrome DevTools Protocol against a user-owned browser session.
- Scope: rendered post text, comments, comment links, and any newly exposed
  attachment controls.

## Guardrails

- No fake accounts, captcha bypass, stealth, or rate evasion.
- Do not print cookie values or credentials.
- Treat comments and LinkedIn-only materials as private research intake.
- Promote no paper/book claims from comments alone.

## Stop Condition

Close each row when logged-in rendered text, visible comments, comment links,
and newly exposed attachments are either captured/read or assigned an explicit
blocker.
