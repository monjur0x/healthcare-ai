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
  review (2026-09-03)") — batches 1-2 done (crew parse bug, encoder
  persistence, torch scaler parity); next candidates: Chroma flag,
  DP return, registry locking

## Next Steps
1. ~~Commit the bug-fix sweep~~ — done (see session log below).
2. Rerun the M3 RAG evaluation with the improved marker-aware queries.
3. Pick P0 review items from .ai/backlog.md for the next fix session.
