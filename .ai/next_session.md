# Next Session

## Objective

Bug-fix sweep against the research proposal is complete and verified.
Commit the changes, then optionally re-run the RAG evaluation to confirm
the marker-aware queries improved retrieval quality.

## Done This Session

- `services.py`: `build_disease_query(prediction, markers)` appends
  elevated markers (from `MARKER_THRESHOLDS`, capped at 5 terms) to the
  disease-anchored RAG query; `assess_risk` now computes
  `max(model P(disease), CREW_RISK_MARKER_WEIGHT * max normalized marker
  elevation)` via `_marker_evidence`, so flagged markers can raise (never
  lower) the score and `risk_factors` never contradict `risk_score`.
- `config.py`: new `RISK_MARKER_WEIGHT` (default 0.5; markers alone cap
  at the medium band). Documented in `backend/.env.example`.
- `tasks.py`: `create_tasks` accepts `features` / `markers` /
  `disease_context` and injects them via `_clinical_context_block` into
  the patient-analysis, disease-prediction, explanation, and
  risk-monitoring task descriptions (fixes LLM narratives ignoring
  clinical values).
- `crew.py`: `_build_query` passes markers; step-3 no longer hits a
  latent `NameError` when the RAG pipeline is absent; duplicate
  `@staticmethod` removed; `run_llm` passes the new task context.
- `crew_logging.py`: rewritten — the old wrapper reassigned
  `crew.kickoff` then called it from inside the wrapper (infinite
  recursion), referenced an undefined `wrap_task_execution` (F821), and
  had duplicated dead bodies (F811) / unused imports (F401).
- `scripts/run_rag_evaluation.py`: metrics are now accumulated and
  aggregated (mean P@1/3/5/10, R@1/3/5/10, MRR) and persisted to
  `artifacts/experiments/rag_evaluation.json` instead of a placeholder
  note; duplicate pipeline init removed. Verified run: MRR 0.49,
  R@10 0.5 (TF-IDF).
- `scripts/ingest_clinical_knowledge.py`: removed duplicated
  output/logging blocks; `ClinicalKnowledgeIngestor` now receives the
  output *directory* (was passed the file path); respects `--output`.
- `api/routes.py`: treatment-planner / explainability agent routes log
  prediction failures instead of silent `except: pass`; module logger
  added.
- Verified: backend-wide `ruff check` + `ruff format --check` clean;
  import smoke OK; deterministic end-to-end analyze (7/7 agents) OK;
  unit-level assertions on the new risk/query/task behavior all pass.

## Next Steps

1. Commit (suggested: `fix(clinical): marker-aware risk & RAG queries,
   LLM task context injection, lint/dead-code cleanup`).
2. Re-run `scripts/run_m3_evaluation.py` (faithfulness) with the
   improved queries to confirm §12 RAG metrics improved.
3. Backlog candidates: dense embedder, corpus expansion.
4. New: full-project review logged to `.ai/backlog.md` ("Full-project
   review (2026-09-03)") as P0/P1/P2 — no fixes applied this session.
   Suggested next fix order: `crew.py:555` parse bug → encoder re-fit →
   torch scaler → Chroma flag → DP return → registry locking.
5. Update 2026-09-03: P0 batch 1 done (crew parse binding + encoder
   persistence chain, verified 7/7 checks + legacy-artifact compat,
   `ruff check` / `ruff format --check` clean). Next: torch scaler,
   Chroma flag, DP return, registry locking.
6. Update 2026-09-04: P0 batch 2 done (torch scaler/encoder parity +
   artifact persistence, verified 5/5 checks incl. legacy torch
   artifact compat, `ruff` clean). Next: Chroma flag, DP return,
   registry locking.
7. Update 2026-09-04: P0 batch 3 done (Chroma flag line removed, probe
   governs; absent/stubbed paths verified 2/2, `ruff` clean). Next:
   DP return, registry locking.
8. Update 2026-09-04: P0 batch 4 done (DP weight sync-back incl.
   prefix-strip + fail-loud mismatch, verified 5/5 through real
   `_train_locally`, `ruff` clean). Next: registry locking.
9. Update 2026-09-04: P0 batch 5 done (SQLite RLock + WAL + atomic
   versioning + duplicate rollback + registry try/finally; 8-thread
   contention verified on all three stores, `ruff` clean). All P0
   closed — P1 next.

## Open Questions

- None blocking.
