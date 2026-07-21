# Andrija Login-Only/API Triage - 2026-05-25

## Recommendation

Do **not** start with LinkedIn's official API for this pass. For third-party
member activity, the official Posts API is permission-limited and is unlikely
to read arbitrary posts/comments. The safest useful route is a visible,
user-owned logged-in browser session, controlled through CDP/Playwright
storage state if available. Do not paste passwords into chat; if cookies are
used, keep them in a local file outside git and treat them as temporary
secrets.

## Priority Counts

- P0: 7
- P1: 6
- P2: 2
- No LinkedIn-cookie value: 1

## P0 Targets

1. Full capture of `7463579851317239808` and `7462756612538073090`.
   These are the only indexed posts with zero useful public content.
2. Logged-in classification of the six newest profile activity IDs without
   public own-post permalinks. These are the best place to discover missing
   current authored posts.
3. Expanded comments/link scan for multi-period PD testing, model-based PD
   rating-scale calibration, WoE encoding instability, and tree-based
   interactions.

## P1 Targets

Scan comments for normal-test reliability, SMI/IFRS9, model shift/MRM and
the two ADSFCR/Working Notes umbrella posts. These can add references or
reviewer-defense language, but most primary assets are already read.

## P2 / Skip

Exact binomial, monobinpy, PDtoolkit, R toolkit, book-release and repo
announcement posts should not consume much session time. They are already
implemented, tooling context, or source-discovery only. The CRC 403 page is
not a LinkedIn blocker and should not be routed through LinkedIn cookies.

## Guardrails

- Use only user-owned visible access; no fake accounts, captcha bypass,
  stealth, or rate evasion.
- Capture only what the account can normally view.
- Stop each target when it is classified, captured, resolved, or shown to
  be non-credit-risk.
- LinkedIn comments remain intake/source-discovery only unless backed by
  an independent source or local result.
