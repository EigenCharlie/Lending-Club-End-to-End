# Profile Remaining Content Audit

Generated: 2026-05-21

## What Is Fully Covered In The Local Corpus

The local corpus covers the prior 59-post index plus 8 LinkedIn child posts
discovered from external links. Within that corpus, every row has a stop
decision in `data/post_execution_decisions.csv`.

Coverage by content type:

- 59 indexed Denis Burakov posts captured from public permalinks.
- 8 LinkedIn child posts associated from resolved external links.
- 28 LinkedIn PDF/document assets captured; 27 extracted with `pdftotext`, 1
  image-based PDF read manually, and parked/low-text decks visually reread.
- 24 image/carousel posts visually read directly through contact sheets or raw
  images.
- 7 text-only posts read from captured post text.
- 109 external link references resolved and classified.
- 21 external links marked as potential evidence snapshotted as readable
  artifacts.
- 13 external LinkedIn post/article links associated or read as child posts.

The analysis intentionally read post text together with its attached image, PDF,
deck, or external link status. Image material was not promoted as bibliography;
it was used for concept intake, backlog decisions, and safe book-language
improvements.

## What Remains Pending Inside The Captured Corpus

There is no missing post-level decision inside the 67-row backlog.

Remaining caveats are artifact-level, not decision-level:

- Raw image assets still record `ocr_tool_missing` because Tesseract was not
  installed. This is a reproducibility gap, not an analysis gap: the image posts
  were reread visually and documented in
  `docs/manual_visual_reread_memo_2026-05-21.md`.
- 43 external links were classified as non-peer-reviewed/contextual sources and
  triaged rather than deeply promoted.
- 32 LinkedIn identity/company/profile links were archived as provenance/context
  rather than read as technical evidence.
- Blocked or rate-limited non-essential links remain labeled in
  `data/external_link_backlog.csv`; none currently controls a book/paper claim.

## What Remains Pending For The Live LinkedIn Profile

The local corpus is not the full live LinkedIn profile. Public LinkedIn pages
observed on 2026-05-21 show the profile/post pages with 141 posts and 16
articles, while the local audited corpus covers 59 indexed posts plus 8 linked
child posts.

Examples of public posts discovered by web search but not present in
`data/posts_index.csv`:

- `7447901245949792256`: NLP, MLE, LLMs, and credit-risk scorecards.
- `7376148352284901376`: Explainable Credit Risk Models workshop deck.
- `7211748798618824704`: trended credit scoring / transactional sequences.
- `7328316894258577408`: Poisson models for event counts in applied risk.

These are pending because they were outside the initial scrape, not because they
were skipped after capture.

## Recommended Next Step

Run a live profile-discovery pass using the authenticated Windows Chrome session:

1. Inventory all visible posts/articles from the profile activity page.
2. Deduplicate against `data/posts_index.csv` and
   `data/external_linkedin_child_post_backlog.csv`.
3. Capture only missing relevant posts first: explainable credit risk, trended
   data, Poisson/count models, NLP/MLE-to-scorecard bridges, AI underwriting, and
   any document/deck-rich posts.
4. Add those missing posts as a second corpus batch with the same columns:
   post text, assets, external links, source status, project destination,
   implementable candidate, and stop condition.

Until that pass is done, the correct status is: captured corpus complete; full
profile incomplete.
