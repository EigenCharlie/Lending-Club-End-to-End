# Paper 4 Notebook B018 Fig.show Patch v410

Generated: 2026-05-17T09:16:51.062782+00:00

v410 applies the explicit `fig.show()` display patch selected by v409.

## Result

- B018 diagnostics: `10` ->
  `0`.
- Global notebook diagnostics: `17` ->
  `7`.
- Changed notebook files: `3`.
- Roundtrip integrity passed: `True`.

## Required Caveat

v410 does not clear remaining F821/SIM/style notebook lint, does not run
post-patch pytest, and does not create Paper 4 final promotion.
