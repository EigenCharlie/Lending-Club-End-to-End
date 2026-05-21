
# Denis Burakov LinkedIn Credit Risk Research Pack

Research pack privado para convertir posts de LinkedIn sobre credit risk,
scorecards, calibracion y MLOps en intake trazable para el libro Quarto, Paper 4
y Paper Estrella.

## Base importada

- `denis_burakov_credit_risk_dossier.md`: dossier inicial con indice, resumen,
  posts relevantes y aplicacion al proyecto.
- `data/posts_index.csv`: indice limpio de 59 posts revisados.
- `data/relevant_posts.csv`: subconjunto de relevancia alta/media.
- `data/external_links_resolved.csv`: enlaces externos detectados y resoluciones
  posibles del scrapeo inicial.
- `attachments/ATTACHMENTS.md`: manifiesto inicial de adjuntos/enlaces por post.

## Capa auditada nueva

Regenerable con:

```bash
python scripts/research/build_linkedin_credit_risk_pack.py --probe-api --api-probe-limit 3 --run-date 2026-05-21
```

Procesamiento de PDFs/imagenes ya capturados:

```bash
python scripts/research/process_linkedin_assets.py
```

Captura publica reproducible de todo el indice previo:

```bash
python scripts/research/capture_linkedin_public_batch.py --post-numbers all --sleep-seconds 0.25
python scripts/research/process_linkedin_assets.py
```

- `data/linkedin_corpus_inventory.csv`: estado por post, captura y completitud.
- `data/attachment_manifest.csv`: adjuntos y enlaces por `asset_id`, con estado
  de OCR/text extraction.
- `data/external_source_log.csv`: enlaces canonicos, tipo de fuente y estado de
  acceso.
- `data/concept_atlas.csv`: conceptos accionables, destino editorial,
  dificultad y riesgo de claim.
- `data/linkedin_api_probe_log.csv`: intento oficial de API y bloqueo/estado.
- `data/human_assisted_capture_queue.csv`: cola segura para captura visible en
  Chrome de Windows.
- `data/public_permalink_capture_log.csv`: resultado de captura publica para los
  59 posts del indice previo.
- `docs/linkedin_claim_evidence_map.md`: mapa post -> concepto -> decision de
  intake.
- `docs/p1_linkedin_reading_memo_2026-05-21.md`: memo analitico del lote P1,
  con conceptos que si pasan a backlog del libro/papers.
- `docs/full_corpus_processing_status_2026-05-21.md`: estado de captura,
  conteos finales y cola restante de lectura/OCR/fuentes externas.
- `docs/overnight_goal_backlog_plan_2026-05-21.md`: reglas de ejecucion,
  condiciones de parada por post y orden de ataque del goal overnight.
- `docs/pdf_text_batch1_execution_memo_2026-05-21.md`: cierre de los posts
  PDF/texto accionables y cambios aplicados.
- `docs/image_batch1_execution_memo_2026-05-21.md`: lectura manual de imagenes
  de alta relevancia.
- `docs/remaining_posts_execution_memo_2026-05-21.md`: cierre de posts de
  prioridad media/baja, texto-only y source-discovery.
- `docs/external_child_posts_execution_memo_2026-05-21.md`: cierre de posts
  LinkedIn externos encontrados al resolver links.
- `docs/external_high_value_sources_memo_2026-05-21.md`: cierre de los 21 links
  externos marcados como evidencia potencial, con snapshots legibles.
- `docs/manual_visual_reread_memo_2026-05-21.md`: relectura visual directa de
  imagenes/carousels y PDFs `park` sin depender de OCR/Tesseract.
- `docs/profile_remaining_content_audit_2026-05-21.md`: separa el corpus local
  ya cerrado de los pendientes del perfil vivo de LinkedIn.
- `data/post_execution_backlog.csv`: backlog maestro por post, con destino,
  implementable posible y condicion de parada.
- `data/external_link_backlog.csv`: resolucion y manejo de las 109 referencias
  externas conservando el post padre.
- `data/high_value_external_source_reading.csv`: estado de lectura y uso
  recomendado para fuentes externas de alto valor.
- `data/post_execution_decisions.csv`: decision final por cada fila del backlog
  maestro.
- `external_sources/readable/`: snapshots legibles de GitHub raw/README,
  preprints y documentacion oficial.
- `attachments/extracted/`: textos extraidos de PDFs/OCR cuando existan assets
  locales en `data/attachment_manifest.csv`.

## Governance

La carpeta puede contener PDFs, imagenes, capturas y OCR como artefactos privados
de investigacion. Ningun claim del libro o papers debe promoverse desde LinkedIn
sin leer el adjunto/fuente canonica completa y etiquetar su estatus
(`peer-reviewed`, `official`, `preprint`, `blog`, `GitHub`, `LinkedIn-only`).
