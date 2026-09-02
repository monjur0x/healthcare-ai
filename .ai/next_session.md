# Next Session

## Objective

Continue the multi-hospital federated build. Phase 1 (distributed Flower
deployment), the Phase 2 federation registry API, the dashboard Federation
panel, the real medical RAG corpus, the n8n doctor-notification branch, the
feedback-driven retrain loop, encrypted gRPC transport, and risk monitoring
on history are complete and verified. Next: commit the risk history work
and pick the next backlog item.

## Done This Session

- Added `backend/risk/` module: `RiskHistorySettings` (`RISK_HISTORY_` env
  prefix: `DB_PATH`, `TREND_WINDOW`, `ESCALATION_THRESHOLD`, `MIN_TREND_POINTS`,
  `ALERTS_ENABLED`), SQLite `RiskHistoryStore`, and schemas
  (`RiskHistoryRecord`, `RiskTrend`, `RiskHistorySummary`, `RiskHistoryResponse`,
  `EscalationAlert`).
- Extended `AnalysisService` with a `risk_history_store` and
  `_persist_risk_history`, called automatically after each `analyze()`
  to record risk score, level, prediction, confidence, and markers.
- Added API routes: `GET /api/v1/risk/history` (all or filtered),
  `GET /api/v1/risk/history/{patient_id}`, `GET /api/v1/risk/trends/{patient_id}`,
  `GET /api/v1/risk/alerts`.
- Trend analysis: linear regression over recent window (`TREND_WINDOW`),
  direction (improving/stable/worsening), slope, average, latest score/level,
  and escalation flag when score delta exceeds `ESCALATION_THRESHOLD`.
- Escalation alerts: active alerts for all patients with score jumps above
  threshold, sorted by timestamp.
- Documented all `RISK_HISTORY_*` variables in `backend/.env.example`.
- Updated README with risk monitoring section and API endpoints table.
- Verified lint (ruff) and format checks pass on all modified files.

## Next Steps

1. Commit + push the risk history work when the user asks.
2. Phase 2+ candidates (backlog):
   - Cross-host deployment docs.
   - Corpus expansion.
   - n8n risk-monitoring workflow (poll alerts and notify).

## Open Questions

- None blocking.
---

## Session: perf(crewai) free-tier tuning — deferred issues

Noticed while wiring LLM execution limits; NOT fixed here (out of scope,
execution-speed-only task):

1. **RAG query-building bug** — `crew.py::_build_query()` builds the
   retrieval query only from `predicted_class` + confidence
   (e.g. "clinical evidence and management for 1 at 58% confidence").
   It never includes patient markers/features or the predicted disease
   NAME, so retrieval quality is degraded for every preset. The
   deterministic Agent-3 path uses this same query.
2. **Risk-scoring issue** — `services.py::assess_risk()` derives risk
   purely from `_positive_class_probability(prediction)` and clamps
   marker-based adjustments; with the current global model the score is
   just the positive-class probability, so `risk_factors` (marker
   thresholds) can disagree with the numeric score (e.g. glucose 210
   flagged but score driven by model confidence).
3. **LLM narrative faithfulness** — ox-alpha run claimed "no vital
   signs / no labs" despite features being present in crew inputs.
   Likely cause: task descriptions embed only `patient.model_dump()`
   (demographics); clinical values reach agents only via base_report
   JSON in context. Consider injecting features/markers into relevant
   task descriptions.

Priority for next session: fix #1 and #3 together (both are prompt/query
content), re-run RAG evaluation afterwards.
