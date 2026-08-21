# Current Context

## Current Milestone

Phase 2 of the multi-hospital federated build: federation model registry
(API + dashboard), real medical RAG corpus, doctor notification via n8n,
feedback-driven retrain loop, encrypted gRPC transport (TLS), and risk
history monitoring with trend analysis and escalation alerts.

## Current Module

`backend/risk/` (persistent risk history store + schemas + trend analysis),
`backend/api/services.py` + `routes.py` + `schemas.py` (risk history
endpoints), `backend/CrewAI/orchestrator/services.py` (tolerant
feature-key alignment).

## Current Task

Risk monitoring on history (complete):

1. `backend/risk/` module: `config.py` (`RiskHistorySettings`, `RISK_HISTORY_`
   prefix: `DB_PATH`, `TREND_WINDOW`, `ESCALATION_THRESHOLD`, `MIN_TREND_POINTS`,
   `ALERTS_ENABLED`), `store.py` (SQLite `RiskHistoryStore`: add /
   get_patient_history / get_recent_scores / get_all_patients /
   compute_trend / get_summary / get_all_summaries / get_escalation_alerts),
   `schemas.py` (`RiskHistoryRecord`, `RiskTrend`, `RiskHistorySummary`,
   `RiskHistoryResponse`, `EscalationAlert`).
2. `AnalysisService` gained `risk_history_store` and `_persist_risk_history`,
   called automatically after each `analyze()` to record risk score, level,
   prediction, confidence, and markers.
3. API routes: `GET /api/v1/risk/history` (all or filtered),
   `GET /api/v1/risk/history/{patient_id}`, `GET /api/v1/risk/trends/{patient_id}`,
   `GET /api/v1/risk/alerts`.
3. Trend analysis: linear regression over recent window; direction
   (improving/stable/worsening), slope, avg, latest; escalation alert when
   score jump exceeds threshold.
4. Escalation alerts: active alerts from all patients, sorted by timestamp.

## Completed

- Phase 1 (committed `ebad74f` + `51d479d`): distributed Flower gRPC
  deployment, hospitals-as-processes, SQLite registry, DP + secure
  aggregation, canonical feature schema.
- Federation registry API (committed `7afa18e`).
- Dashboard federation panel (committed `b351e0e`).
- Bundled medical corpus authored (committed `db8708e`).
- Doctor notification via n8n (committed `1a9b500`).
- Feedback / retrain loop (committed `dfd8075`).
- Encrypted gRPC transport (committed `06a57aa`).
- Risk monitoring on history (this task; not yet committed).

## Next Files (optional / backlog)

- Cross-host deployment docs.
- Corpus expansion.
- n8n risk-monitoring workflow.

## Design Notes

- n8n 2.x executes the *active/published* workflow version
  (`workflow_history` row referenced by `workflow_entity.activeVersionId`),
  not the draft in `workflow_entity`. Import with
  `n8n import:workflow --input=<list-with-id>` (the importer needs a JSON
  array with an explicit `id`), then set `active=1` + `activeVersionId` to
  a new `workflow_history` row (`versionId` is the PK, no `id` column),
  then restart n8n.
- The retrained artifact replaces the served model: `retrain_from_feedback`
  writes to `artifacts/<preset>/global_model.joblib` and sets
  `active_preset`, so no restart is needed to redeploy.
- Feedback features are stored raw (as submitted) and re-keyed against the
  base dataset columns on retrain; the augmented CSV uses the pipeline
  column convention so `prepare_tabular_data` treats them like base rows.
- The feedback store defaults to `artifacts/feedback.db` (a fresh temp DB
  when the service builds it under an ephemeral artifacts dir).
- TLS for Flower gRPC uses standard PEM certificates; server requires
  CA + cert + key; client requires CA (for verification) and optionally
  client cert + key for mutual TLS.
- Risk history is persisted per-analysis with patient_id, preset, score,
  level, prediction, confidence, markers; trend uses linear regression
  over the configured window; escalation alert triggers on score delta
  exceeding threshold.

## Status

Risk monitoring on history complete. Lint (ruff) and format checks pass
on all modified files. Repo not yet committed (user drives commits).