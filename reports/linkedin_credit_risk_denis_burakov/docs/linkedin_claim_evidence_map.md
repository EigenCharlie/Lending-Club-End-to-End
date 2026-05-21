# LinkedIn Credit Risk Claim-Evidence Map: Denis Burakov

Generated: 2026-05-21

## Executive Status

This pack reprocesses the prior 59-post index into an auditable intake surface.
All 59 posts from the prior index have now been captured from public permalinks.
The P1 batch has also been read into a first analytic memo, and the remaining
indexed posts are closed through batch execution memos with explicit
append/park/archive decisions.

- Posts reprocessed: 59
- Relevance mix: Alta: 39, Baja: 6, Media: 14
- Attachment mix: Image/carousel: 27, LinkedIn document/deck: 28, Video: 1, none_recorded: 3
- Official API probe status: blocked_unauthenticated_no_oauth_token: 9
- Public permalink captures: 59 posts; 28 LinkedIn document PDFs/transcripts; 77 feedshare image URLs detected
- Manual image/PDF visual reads completed: 17 assets from the P1 batch
- OCR-independent visual reread completed: 24 image/carousel posts plus 10
  parked/low-text PDF decks
- External link references resolved with handling decisions: 109
- High-value external sources snapshotted as readable artifacts: 21 / 21
- Post backlog closure: 67 / 67 rows closed in `data/post_execution_decisions.csv`
- Governance rule: LinkedIn material is intake evidence only until the attached
  PDF/image/deck/external source is read and source status is labeled.

## Full Public Capture Update

The public permalink capture pass covers all 59 posts in the prior scrape.
Local HTML/text, document PDFs/transcripts where LinkedIn exposed them, image
files for image posts, and the capture log are stored under the pack. The P1
reading memo remains the first deep analytic synthesis; later batches are closed
through targeted execution memos and explicit append/park/archive decisions.
The 21 external links marked as potential evidence now have readable local
snapshots and source-level stop decisions in
`docs/external_high_value_sources_memo_2026-05-21.md`.
The later manual visual reread is recorded in
`docs/manual_visual_reread_memo_2026-05-21.md`; it promoted only three
book-language changes and did not add bibliography claims from images.

## P1 Reading Update

The first priority batch covers posts 1, 3, 4, 5, 6, 8, 14, 15, 20, 22, 31, 32,
35, 45, 55, and 58. Public permalink capture produced local HTML/text, document
PDFs/transcripts where LinkedIn exposed them, image files for image posts, and a
capture log under `data/public_permalink_capture_log.csv`. The analytic synthesis
is in `docs/p1_linkedin_reading_memo_2026-05-21.md`.

## API And Capture Decision

The implementation records the official API path first in
`data/linkedin_api_probe_log.csv`. When no OAuth token is supplied, the probe is
sent without an Authorization header and should return a 401 authorization
blocker. If a token is supplied later, the script will attempt the documented
Posts API endpoint for activity/share/UGC URN variants. Current working
assumption remains that third-party member post retrieval is blocked by
restricted `r_member_social` access, so the approved fallback is a visible,
user-owned, human-assisted Chrome workflow.

## Concept Atlas

| Concept | Family | Posts | Destination | Decision | Claim Risk |
| --- | --- | --- | --- | --- | --- |
| Credit-risk MLOps with MLflow/SageMaker/LocalStack | MLOps and reproducibility | 9 | Book Ch10; implementation companion | append_to_atlas_after_source_verification | Low for engineering appendix; high if framed as research contribution. |
| GBDT leaf WOE and boosted scorecards | Interpretable ML / scorecards | 7 | Book Ch05/Ch06; future benchmark | append_to_atlas_after_source_verification | High: avoid adding dependency-heavy benchmark without gate. |
| Probabilistic LGD via quantile or multiclass bins | Risk parameter uncertainty | 6 | Book Ch07/Ch10; Paper4 IFRS/LGD appendix | append_to_atlas_after_source_verification | High: project data limitations and ECL proxy boundaries apply. |
| External resource collections and On Credit book trail | Source discovery | 6 | Bibliography triage; Book foundations | append_to_atlas_after_source_verification | Medium: source status must be labeled; not all resources are peer reviewed. |
| Brier vs Gini/ECE separation | Calibration and metric governance | 5 | Book Ch06; Paper4 metric governance; Paper Estrella reviewer defense | append_to_atlas_after_source_verification | Medium: requires validated source beyond LinkedIn before manuscript claim. |
| SHAP distillation into scorecard-style explanations | Explainability and model governance | 5 | Book Ch06/Ch10; Paper4 governance appendix | append_to_atlas_after_source_verification | Medium: explanation of predictions is not causal mechanism. |
| Somers D / Dxy for ordinal or continuous outcomes | Credit-risk metrics | 3 | Book Ch06; possible LGD metric extension | append_to_atlas_after_source_verification | Low: as metric addition, not a core contribution. |
| Economic value of Gini and acceptance-rate framing | Credit decision economics | 2 | Book Ch09; Paper Estrella introduction/discussion | append_to_atlas_after_source_verification | Medium: must tie to project artifacts, not generic LinkedIn examples. |
| Class imbalance and probability correction | Calibration under rare events | 2 | Book Ch06 calibration caveats | append_to_atlas_after_source_verification | Low-medium: align with existing calibration evidence. |
| Classification/probability intervals and Pearsonify contrast | Probability uncertainty | 2 | Book Ch07; Paper Estrella related-work contrast | append_to_atlas_after_source_verification | Medium: distinguish classical intervals, Venn-Abers, and conformal intervals. |
| Observation-level Gini contribution diagnostics | Ranking diagnostics | 1 | Book Ch06; Paper4 appendix diagnostic | append_to_atlas_after_source_verification | Medium: useful as diagnostic, not a new guarantee. |
| WOE recalibration under drift | Scorecard maintenance | 1 | Book Ch05; Paper4 bounded prototype candidate | append_or_prototype_only_after_source_verification | Medium-high: needs reproducible Lending Club drift experiment. |
| Fine-tuning GBDT when new data sources arrive | Model maintenance | 1 | Book Ch10; Paper4 drift/maintenance candidate | append_or_prototype_only_after_source_verification | Medium-high: needs stable train/update protocol. |
| Robust logistic regression for noisy labels/outliers | Robust statistics | 1 | Book Ch06 baseline robustness note | append_to_atlas_after_source_verification | Medium: baseline only unless empirically material. |

## Post-Level Claim Intake

| # | Activity | Rel | Theme | Attachment | Decision | Destination | Source Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 7458410505825685505 | Alta | Calibracion y seleccion de modelos | Image/carousel | append_candidate_after_source_verification | Book Ch06; Paper4 metric governance | p1_memo_completed_pending_external_sources |
| 2 | 7456960883634663424 | Alta | Model maintenance y nuevas variables | Image/carousel | append_candidate_after_source_verification | Research archive / source discovery | public_permalink_captured_pending_asset_reading_or_ocr |
| 3 | 7454786558127161344 | Alta | Metricas de ranking | LinkedIn document/deck | append_candidate_after_source_verification | Book Ch06; Paper4 metric governance | p1_memo_completed_pending_external_sources |
| 4 | 7452612214328537088 | Alta | Economia de Gini y distribuciones score | Image/carousel | append_candidate_after_source_verification | Book Ch06; Paper4 metric governance; Book Ch09; Paper Estrella reviewer defense | p1_memo_completed_pending_external_sources |
| 5 | 7449713105552482304 | Alta | WOE recalibration | Image/carousel | append_candidate_after_source_verification | Book Ch05/Ch06; Book Ch06; Paper4 metric governance | p1_memo_completed_pending_external_sources |
| 6 | 7439581405879103488 | Alta | Debug de Gini | Image/carousel | append_candidate_after_source_verification | Book Ch06; Paper4 metric governance | p1_memo_completed_pending_external_sources |
| 7 | 7437407111698984960 | Alta | Libro y fundamentos | Image/carousel | append_candidate_after_source_verification | Book Ch05/Ch06 | public_permalink_captured_pending_asset_reading_or_ocr |
| 8 | 7436682459037089793 | Alta | Scorecards con interacciones GBDT | LinkedIn document/deck | append_candidate_after_source_verification | Book Ch05/Ch06 | p1_memo_completed_pending_external_sources |
| 9 | 7435957418662207488 | Baja | Certificacion GenAI/AWS | Image/carousel | archive_low_priority | Book Ch10; implementation companion | public_permalink_captured_pending_asset_reading_or_ocr |
| 10 | 7435378973288685568 | Media | Calidad de datos y adopcion analitica | Image/carousel | park_or_context_after_source_verification | Book Ch10; implementation companion | public_permalink_captured_pending_asset_reading_or_ocr |
| 11 | 7434266280364498944 | Alta | Libro On Credit | Image/carousel | append_candidate_after_source_verification | Research archive / source discovery | public_permalink_captured_pending_asset_reading_or_ocr |
| 12 | 7431246513466609664 | Alta | Codigo y libro On Credit | Image/carousel | append_candidate_after_source_verification | Research archive / source discovery | public_permalink_captured_pending_asset_reading_or_ocr |
| 13 | 7429434643583725571 | Alta | Gini en datos desbalanceados | Image/carousel | append_candidate_after_source_verification | Book Ch06; Paper4 metric governance | public_permalink_captured_pending_asset_reading_or_ocr |
| 14 | 7426897722655277074 | Alta | LGD probabilistico | LinkedIn document/deck | append_candidate_after_source_verification | Book Ch07; Paper Estrella framing; Book Ch07/Ch10; Paper4 IFRS/LGD appendix | p1_memo_completed_pending_external_sources |
| 15 | 7423998736009375745 | Alta | Valor economico de Gini | Image/carousel | append_candidate_after_source_verification | Book Ch06; Paper4 metric governance; Book Ch09; Paper Estrella reviewer defense | p1_memo_completed_pending_external_sources |
| 16 | 7421824489165979649 | Alta | MLOps credit risk en AWS | LinkedIn document/deck | append_candidate_after_source_verification | Book Ch10; implementation companion | public_permalink_captured_pending_asset_reading_or_ocr |
| 17 | 7421223494132314112 | Media | Testing estadistico/A-B | Image/carousel | park_or_context_after_source_verification | Research archive / source discovery | public_permalink_captured_pending_asset_reading_or_ocr |
| 18 | 7419287581689008128 | Alta | AWS/credit risk guide | LinkedIn document/deck | append_candidate_after_source_verification | Book Ch10; implementation companion | public_permalink_captured_pending_asset_reading_or_ocr |
| 19 | 7417475730277670912 | Media | Fisher exact test | Image/carousel | park_or_context_after_source_verification | Research archive / source discovery | public_permalink_captured_pending_asset_reading_or_ocr |
| 20 | 7416750932140552192 | Alta | FastWoe | LinkedIn document/deck | append_candidate_after_source_verification | Book Ch05/Ch06; Book Ch06; Paper4 metric governance | p1_memo_completed_pending_external_sources |
| 21 | 7413851823381598208 | Alta | Credit Risk Modeling on AWS | Image/carousel | append_candidate_after_source_verification | Book Ch10; implementation companion | public_permalink_captured_pending_asset_reading_or_ocr |
| 22 | 7410227928887783424 | Alta | SHAP y scorecards | LinkedIn document/deck | append_candidate_after_source_verification | Book Ch05/Ch06; Book Ch06/Ch10; Paper4 governance appendix | p1_memo_completed_pending_external_sources |
| 23 | 7407691340332834816 | Media | LLM/AI notes | LinkedIn document/deck | park_or_context_after_source_verification | Research archive / source discovery | public_permalink_captured_pending_asset_reading_or_ocr |
| 24 | 7404429791552270336 | Alta | Boosted scorecards | Image/carousel | append_candidate_after_source_verification | Book Ch05/Ch06 | public_permalink_captured_pending_asset_reading_or_ocr |
| 25 | 7401530693182427137 | Media | SageMaker pipelines | none_recorded | park_or_context_after_source_verification | Book Ch10; implementation companion | public_permalink_captured_pending_asset_reading_or_ocr |
| 26 | 7400081067224743936 | Alta | Scoring en AWS | LinkedIn document/deck | append_candidate_after_source_verification | Book Ch10; implementation companion | public_permalink_captured_pending_asset_reading_or_ocr |
| 27 | 7397906690391601154 | Media | Deep learning en credit risk | LinkedIn document/deck | park_or_context_after_source_verification | Book Ch07; Paper Estrella framing | public_permalink_captured_pending_asset_reading_or_ocr |
| 28 | 7396457171279970304 | Alta | xBooster/FastWoe updates | Image/carousel | append_candidate_after_source_verification | Book Ch05/Ch06 | public_permalink_captured_pending_asset_reading_or_ocr |
| 29 | 7394653161082314752 | Media | Local AWS dev | none_recorded | park_or_context_after_source_verification | Book Ch10; implementation companion | public_permalink_captured_pending_asset_reading_or_ocr |
| 30 | 7392833273737138176 | Alta | Logistic regression foundations | LinkedIn document/deck | append_candidate_after_source_verification | Book Ch05/Ch06; Book Ch06; Paper4 metric governance | public_permalink_captured_pending_asset_reading_or_ocr |
| 31 | 7391383804583645184 | Alta | Multiclass WOE | LinkedIn document/deck | append_candidate_after_source_verification | Book Ch05/Ch06; Book Ch07/Ch10; Paper4 IFRS/LGD appendix | p1_memo_completed_pending_external_sources |
| 32 | 7389209473522999296 | Alta | SHAP distillation | LinkedIn document/deck | append_candidate_after_source_verification | Book Ch06/Ch10; Paper4 governance appendix | p1_memo_completed_pending_external_sources |
| 33 | 7387020037041119234 | Alta | CatBoost lifecycle | none_recorded | append_candidate_after_source_verification | Book Ch10; implementation companion; Book Ch06/Ch10; Paper4 governance appendix | public_permalink_captured_pending_asset_reading_or_ocr |
| 34 | 7384120953489854464 | Alta | WOE foundations | LinkedIn document/deck | append_candidate_after_source_verification | Book Ch05/Ch06 | public_permalink_captured_pending_asset_reading_or_ocr |
| 35 | 7363453428431224832 | Alta | Class imbalance y calibracion | LinkedIn document/deck | append_candidate_after_source_verification | Book Ch06; Paper4 metric governance | p1_memo_completed_pending_external_sources |
| 36 | 7361290473417633792 | Baja | Fisher-Yates shuffle | Video | archive_low_priority | Research archive / source discovery | public_permalink_captured_pending_asset_reading_or_ocr |
| 37 | 7358383766534316032 | Media | Fraud digital lending | Image/carousel | park_or_context_after_source_verification | Research archive / source discovery | public_permalink_captured_pending_asset_reading_or_ocr |
| 38 | 7356587000176533504 | Baja | Visual generation | Image/carousel | archive_low_priority | Research archive / source discovery | public_permalink_captured_pending_asset_reading_or_ocr |
| 39 | 7353684060264632320 | Alta | Robust logistic regression | LinkedIn document/deck | append_candidate_after_source_verification | Book Ch05/Ch06 | public_permalink_captured_pending_asset_reading_or_ocr |
| 40 | 7351513545626312704 | Alta | Resource collection | LinkedIn document/deck | append_candidate_after_source_verification | Research archive / source discovery | public_permalink_captured_pending_asset_reading_or_ocr |
| 41 | 7350773655057960960 | Media | MCMC/logistic regression | Image/carousel | park_or_context_after_source_verification | Book Ch05/Ch06; Book Ch07; Paper Estrella framing | public_permalink_captured_pending_asset_reading_or_ocr |
| 42 | 7348984398186229761 | Alta | Precision/recall y prevalencia | LinkedIn document/deck | append_candidate_after_source_verification | Book Ch06; Paper4 metric governance | public_permalink_captured_pending_asset_reading_or_ocr |
| 43 | 7346840312377421826 | Baja | AWS certification | Image/carousel | archive_low_priority | Book Ch10; implementation companion | public_permalink_captured_pending_asset_reading_or_ocr |
| 44 | 7302591232625541122 | Alta | Explainability and scorecards | LinkedIn document/deck | append_candidate_after_source_verification | Book Ch05/Ch06; Book Ch06/Ch10; Paper4 governance appendix | public_permalink_captured_pending_asset_reading_or_ocr |
| 45 | 7297638535648481280 | Alta | Conformal/probability intervals | LinkedIn document/deck | append_candidate_after_source_verification | Book Ch07; Paper Estrella framing | p1_memo_completed_pending_external_sources |
| 46 | 7290400805109526529 | Alta | WOE to logistic regression | LinkedIn document/deck | append_candidate_after_source_verification | Book Ch05/Ch06 | public_permalink_captured_pending_asset_reading_or_ocr |
| 47 | 7287502960530518016 | Media | Random Forest compression | Image/carousel | park_or_context_after_source_verification | Research archive / source discovery | public_permalink_captured_pending_asset_reading_or_ocr |
| 48 | 7282429540482568193 | Alta | Logistic confidence intervals | LinkedIn document/deck | append_candidate_after_source_verification | Book Ch05/Ch06; Book Ch07; Paper Estrella framing | public_permalink_captured_pending_asset_reading_or_ocr |
| 49 | 7272626194091552768 | Alta | WoeBoost | Image/carousel | append_candidate_after_source_verification | Book Ch05/Ch06 | public_permalink_captured_pending_asset_reading_or_ocr |
| 50 | 7259829440568836097 | Baja | GANs | Image/carousel | archive_low_priority | Research archive / source discovery | public_permalink_captured_pending_asset_reading_or_ocr |
| 51 | 7254740864336285696 | Baja | WOE for image generation | Image/carousel | archive_low_priority | Book Ch05/Ch06 | public_permalink_captured_pending_asset_reading_or_ocr |
| 52 | 7252204203853443072 | Media | Bias-variance | LinkedIn document/deck | park_or_context_after_source_verification | Research archive / source discovery | public_permalink_captured_pending_asset_reading_or_ocr |
| 53 | 7247130806010187776 | Media | Boosting beyond trees | LinkedIn document/deck | park_or_context_after_source_verification | Research archive / source discovery | public_permalink_captured_pending_asset_reading_or_ocr |
| 54 | 7242057402705993728 | Alta | Binary vs multiclass credit default | LinkedIn document/deck | append_candidate_after_source_verification | Book Ch07/Ch10; Paper4 IFRS/LGD appendix | public_permalink_captured_pending_asset_reading_or_ocr |
| 55 | 7239520617035640833 | Alta | Calibration toolkit | LinkedIn document/deck | append_candidate_after_source_verification | Book Ch06; Paper4 metric governance | p1_memo_completed_pending_external_sources |
| 56 | 7234084801039790080 | Media | WOE text classification | Image/carousel | park_or_context_after_source_verification | Book Ch05/Ch06 | public_permalink_captured_pending_asset_reading_or_ocr |
| 57 | 7229369908193579009 | Media | Visualizing gradient boosting | Image/carousel | park_or_context_after_source_verification | Research archive / source discovery | public_permalink_captured_pending_asset_reading_or_ocr |
| 58 | 7226833197983006722 | Alta | Somers D / Dxy | LinkedIn document/deck | append_candidate_after_source_verification | Book Ch06; Paper4 metric governance; Book Ch07/Ch10; Paper4 IFRS/LGD appendix | p1_memo_completed_pending_external_sources |
| 59 | 7224433718059274241 | Alta | Fisher Scoring package | LinkedIn document/deck | append_candidate_after_source_verification | Research archive / source discovery | public_permalink_captured_pending_asset_reading_or_ocr |

## Source Status Rules

- `append_candidate_after_source_verification`: relevant to the project, but not
  manuscript evidence until full post and attachment/source content are read.
- `park_or_context_after_source_verification`: useful context or future lane,
  but not first-wave Quarto/Paper material.
- `archive_low_priority`: keep permalink and summary only unless a reviewer or
  later source trail makes it relevant.

## Official/Workflow References Used For Feasibility

- https://learn.microsoft.com/en-us/linkedin/marketing/community-management/shares/posts-api
- https://learn.microsoft.com/en-us/linkedin/marketing/community-management/community-management-overview
- https://www.linkedin.com/help/linkedin/answer/a1341387
- https://www.linkedin.com/legal/crawling-terms
- https://playwright.dev/python/docs/auth
- https://playwright.dev/python/docs/api/class-download
- https://playwright.dev/python/docs/screenshots
- https://chromedevtools.github.io/devtools-protocol/

## Files In This Pack

- `data/linkedin_corpus_inventory.csv`
- `data/attachment_manifest.csv`
- `data/external_source_log.csv`
- `data/external_link_backlog.csv`
- `data/high_value_external_source_reading.csv`
- `data/concept_atlas.csv`
- `data/linkedin_api_probe_log.csv`
- `data/human_assisted_capture_queue.csv`
- `data/public_permalink_capture_log.csv`
- `docs/linkedin_claim_evidence_map.md`
- `docs/p1_linkedin_reading_memo_2026-05-21.md`
- `docs/external_high_value_sources_memo_2026-05-21.md`
- `docs/manual_visual_reread_memo_2026-05-21.md`
- `docs/full_corpus_processing_status_2026-05-21.md`
