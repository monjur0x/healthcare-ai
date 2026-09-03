# Current Context

## Current Milestone
M5 — Bug-fix sweep against the research proposal (deferred clinical issues)

## Current Module
`backend/CrewAI/orchestrator/`, `backend/api/`, `backend/scripts/`

## Current Task
- ✅ Bug-fix sweep completed (see .ai/backlog.md "Deferred issues" for the list)
- ✅ Full-project review completed — findings logged to .ai/backlog.md
  "Full-project review (2026-09-03)" as P0/P1/P2 (no fixes applied)
- ✅ P0 batch 1 completed: crew `_parse_report` binding fix + encoder
  mapping persistence (params/from_params, transformer/pipeline/model/
  API wiring, unseen-category fail-loud)
- ✅ P0 batch 2 completed: torch scaler/encoder parity
  (`TorchMLPClassifier` params interface + artifact persistence,
  fixes `analyze_csv` / crew-scaler `AttributeError` for served torch
  models)
- ✅ P0 batch 3 completed: Chroma availability-flag fix (import probe
  governs again; both absent/stubbed paths verified)
- ✅ P0 batch 4 completed: DP-trained weights sync back into the client
  model (`_apply_trained_weights` with prefix-strip + fail-loud
  mismatch; DP/non-DP paths verified 5/5)
- ✅ P0 batch 5 completed: SQLite locking across registry + risk +
  feedback stores (RLock, WAL, atomic versioning, duplicate-round
  rollback, registry try/finally; 8-thread contention verified)
- ✅ P1 batch 1 completed: FedAvg count-weighting on both servers with
  secure/non-secure parity (optional counts, pre-scaled masked mean;
  verified identical globals)

## Completed
- ✅ M3.1: Privacy layer (anonymize_frame) wired into hospital data loading
- ✅ M3.2: RAG evaluation with 18 clinical queries
- ✅ M4: CrewAI logging enhancement (agent execution logging)
- ✅ M3.4: n8n risk-monitoring + clinical-full workflows
- ✅ Bug sweep: RAG query markers, marker-aware risk score, LLM task
  context injection, crew_logging.py recursion, RAG evaluation metrics
  aggregation, silent-exception logging in agent routes

## Remaining Backlog
- Dense embedder for better RAG faithfulness (sentence-transformers)
- Corpus expansion
- Six-agent pipeline as distinct CrewAI agents in run_llm
- Full-project review P0/P1/P2 items (see .ai/backlog.md "Full-project
  review (2026-09-03)") — all P0 batches done (crew parse bug, encoder
  persistence, torch scaler parity, Chroma flag, DP return, registry
  locking); P1 batch 1 done (FedAvg weighting); P1 next (DP accounting,
  OTP nonce, canonical mappings, agent metrics, API hardening,
  risk/feedback semantics, baseline rigor)

## Next Steps
1. ~~Commit the bug-fix sweep~~ — done (see session log below).
2. Rerun the M3 RAG evaluation with the improved marker-aware queries.
3. Pick P0 review items from .ai/backlog.md for the next fix session.
