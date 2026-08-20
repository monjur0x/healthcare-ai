# Current Context

## Current Milestone

Phase 2 of the multi-hospital federated build: surface the federation
model registry through the FastAPI and the Streamlit dashboard, replace
the placeholder RAG corpus with a real medical corpus, and add a
doctor-notification branch to the n8n orchestration for high-risk
clinical analyses.

## Current Module

`n8n/healthcare-endtoend.json` (high-risk notification branch) and
`backend/CrewAI/orchestrator/services.py` + `tools.py` (positive-class
risk scoring).

## Current Task

Doctor notification via n8n (complete):

1. `n8n/healthcare-endtoend.json` extended with an `IF: High Risk?`
   branch after `IF: Analysis Succeeded?`: high-risk analyses fire an
   `HTTP: Notify Doctor` webhook to the `DOCTOR_NOTIFY_WEBHOOK` n8n env
   var, then a `Code: Pass Report After Notify` node restores the
   clinical report so the webhook response is unchanged. The notify is
   best-effort (`onError: continueErrorOutput`).
2. `assess_risk` fixed to score the probability of the positive
   (disease) class instead of max-class confidence, so a confident
   prediction of the healthy class scores *low* risk. Helper
   `_positive_class_probability`; `RiskAssessmentTool` description
   updated.
3. Workflow URLs switched from `localhost` to `127.0.0.1` so the n8n
   HTTP nodes deterministically reach the FastAPI over IPv4.
4. Live verification (high-risk patient → notification fires + full
   report returned; low-risk patient → no notification):
   - healthy → `risk low` (0.0003), no notify;
   - sick → `risk high` (0.9994), notify fires with alert payload.

## Completed

- Phase 1 (committed `ebad74f` + `51d479d`): distributed Flower gRPC
  deployment, hospitals-as-processes, SQLite registry, DP + secure
  aggregation, canonical feature schema.
- Federation registry API (committed `7afa18e`).
- Dashboard federation panel (committed `b351e0e`).
- Bundled medical corpus authored (committed `db8708e`).
- Doctor notification via n8n (this task; not yet committed).

## Next Files (optional / backlog)

- Feedback / retrain loop.
- Encrypted gRPC transport.
- Risk monitoring on history.
- Cross-host deployment docs.
- Corpus expansion.

## Design Notes

- n8n 2.x executes the *active/published* workflow version
  (`workflow_history` row referenced by `workflow_entity.activeVersionId`),
  not the draft in `workflow_entity`. Direct DB edits to the draft do not
  affect runs; use `n8n import:workflow` (or update both the history row
  and `activeVersionId`) and then activate.
- n8n blocks arbitrary env access in node expressions by default; set
  `N8N_BLOCK_ENV_ACCESS_IN_NODE=false` so `$env.DOCTOR_NOTIFY_WEBHOOK`
  resolves.
- `DOCTOR_NOTIFY_WEBHOOK` is intentionally not hardcoded: the repo JSON
  keeps it an env var so deployments can point it at a pager/chat-bot or
  hospital notification service without editing the workflow.
- The `HTTP: Notify Doctor` node's own output is discarded via the
  pass-through node so the webhook response still carries the clinical
  report (the notify receiver returns `{"received":true}`).

## Status

Doctor notification via n8n complete and verified end-to-end against the
live n8n + FastAPI. Repo not yet committed (user drives commits).