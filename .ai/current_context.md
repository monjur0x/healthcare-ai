# Current Context

## Current Milestone
M5 — Bug-fix sweep against the research proposal (deferred clinical issues)

## Current Module
`backend/CrewAI/orchestrator/`, `backend/api/`, `backend/scripts/`

## Current Task
- ✅ Bug-fix sweep completed (see .ai/backlog.md "Deferred issues" for the list)
- ✅ n8n consolidation: retired `clinical-full` v1 + `clinical-analysis`
  to `n8n/archive/` (with README); canonical set is now
  `healthcare-endtoend`, `clinical-full-v2`, `risk-monitoring`,
  `feedback-retrain`. Demo console, `start_demo.py`, README, and
  RUN_GUIDE retargeted to `clinical-full-v2` (response keys verified
  compatible; rejection expectation updated).
- ✅ Full-project review completed — findings logged to .ai/backlog.md
  "Full-project review (2026-09-03)" as P0/P1/P2 (no fixes applied)
- ✅ P0 batch 1 completed: crew `_parse_report` binding fix + encoder
  mapping persistence (params/from_params, transformer/pipeline/model/
  API wiring, unseen-category fail-loud)
- ✅ CrewAI rebuilt lean (user call): 5 agents × 5 sequential tasks,
  max_iter 1, max_rpm pacing, short prompts; dropped analyst +
  explainer agents (folded into predictor/report tasks) and the dead
  logging shim; retry/backoff + `llm_path` hardening kept. Verified:
  ruff clean, 5/5 wiring in serving env, deterministic run unchanged
  (high 0.994, 0.04s). Live LLM proof awaits a working key.
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
- ✅ P1 batch 2 completed: per-round worst-case epsilon accounting on
  both servers (no more ×N overcount; verified end-to-end)
- ✅ P1 batch 3 completed: OTP seed+round-bound masks, canonical
  bu-fix + label/mapping hygiene, agent-metrics payload visibility,
  full LLM tool wiring, API service lock + subprocess timeout +
  route delegation with fallback flags, risk trend/ALERTS/n8n-dedup,
  feedback consumed visibility + guarded consume, baseline rigor
  (M2/M3/privacy scripts), RAG ground-truth/metrics/threshold/
  BGE/refit fixes — all verified incl. live M3 + RAG eval runs

## Completed
- ✅ M3.1: Privacy layer (anonymize_frame) wired into hospital data loading
- ✅ M3.2: RAG evaluation with 18 clinical queries
- ✅ M4: CrewAI logging enhancement (agent execution logging)
- ✅ M3.4: n8n risk-monitoring + clinical-full workflows
- ✅ Bug sweep: RAG query markers, marker-aware risk score, LLM task
  context injection, crew_logging.py recursion, RAG evaluation metrics
  aggregation, silent-exception logging in agent routes

## Remaining Backlog
- Dense embedder for better RAG faithfulness (sentence-transformers;
  blocked on installing the optional dep + model download — code path
  ready: `RAG_EMBEDDING_MODEL=sentence-transformer`)
- Corpus expansion (add topic docs so eval ground truth maps 1:1;
  deferred: new clinical summaries need expert review before merging)
- ~~Six-agent pipeline as distinct CrewAI agents in run_llm~~ — done:
  stale item, `run_llm` already builds 7 distinct agents × 7 tasks
  with tools (verified `AGENT_PROFILES` + `create_tasks` keys)
- Full-project review items (see .ai/backlog.md) — P0 closed, P1
  closed, P2 closed except explicitly scoped-out multimodal research
  (datasets + architectures + training pipeline). Next: verify live
  system + commit.

## Next Steps
1. ~~Commit the bug-fix sweep~~ — done (see session log below).
2. Rerun the M3 RAG evaluation with the improved marker-aware queries.
3. Pick P0 review items from .ai/backlog.md for the next fix session.
