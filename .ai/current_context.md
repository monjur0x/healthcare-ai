# Current Context

## Current Milestone

Phase 2 of the multi-hospital federated build: federation model registry
(API + dashboard), real medical RAG corpus, doctor notification via n8n,
and the feedback-driven retrain loop (n8n → retrain → redeploy when
pending clinician feedback crosses a threshold).

## Current Module

`backend/feedback/` (persistent clinician-feedback store + schemas),
`backend/api/services.py` + `routes.py` + `schemas.py` (feedback
endpoints + retrain service methods), `backend/CrewAI/orchestrator/
services.py` (tolerant feature-key alignment), and `n8n/feedback-retrain
.json` (orchestration workflow).

## Current Task

Feedback / retrain loop (complete):

1. `backend/feedback/` module: `config.py` (`FeedbackSettings`, `FEEDBACK_`
   prefix: `DB_PATH`, `RETRAIN_THRESHOLD`, `RETRAIN_ENABLED`), `store.py`
   (SQLite `FeedbackStore`: add / get / list_pending / count_pending /
   count_total / recent / mark_consumed), `schemas.py` (`FeedbackRequest`,
   `FeedbackRecord`, `FeedbackSummary`, `FeedbackStatus`). API-level
   `RetrainRequest` / `RetrainResponse` live in `api/schemas.py` to avoid a
   feedback → api import cycle.
2. `AnalysisService` gained `feedback_store`, `record_feedback`,
   `feedback_status`, `retrain_from_feedback` (returns `RetrainResult`
   dataclass), `_validate_preset`, `_write_augmented_dataset`. Retrain
   builds an augmented CSV (base dataset + pending feedback rows, target =
   confirmed label), calls the existing `train()` with `dataset` overriding
   the preset file (`_resolve_dataset` now lets an explicit path win while
   the preset supplies target + output dir), marks consumed rows, and
   serves the new model immediately (`active_preset` set).
3. API routes: `POST /api/v1/feedback`, `GET /api/v1/feedback/status`,
   `POST /api/v1/feedback/retrain`.
4. `n8n/feedback-retrain.json` (webhook `feedback-retrain`): GET feedback
   status → resolve preset request → `IF: Ready to Retrain?` →
   `HTTP: Retrain Model` → build success/not-ready/error responses.
5. Feature-key alignment fix in `run_prediction`: retrained models carry
   pipeline-normalized `feature_names` (`bloodpressure`,
   `diabetespedigreefunction`) that did not match the snake_case keys the
   manual / n8n path sends (`blood_pressure`,
   `diabetes_pedigree_function`). Added `_align_feature_keys` +
   `_normalize_feature_key` so both spellings resolve to the model's names.
6. Live verification (all end-to-end):
   - `POST /api/v1/feedback` × 5 → status `pending: 5, ready: true`;
   - `POST /api/v1/feedback/retrain` → accuracy 0.8187, consumed 5,
     served model replaced (`active_preset = diabetes`);
   - `POST /webhook/feedback-retrain` → `status: success, consumed: 5`,
     and `not_ready` when pending is below threshold;
   - endtoend still works with the retrained model: low-risk patient
     (risk low, no notify), high-risk patient (risk high, notify fires).

## Completed

- Phase 1 (committed `ebad74f` + `51d479d`): distributed Flower gRPC
  deployment, hospitals-as-processes, SQLite registry, DP + secure
  aggregation, canonical feature schema.
- Federation registry API (committed `7afa18e`).
- Dashboard federation panel (committed `b351e0e`).
- Bundled medical corpus authored (committed `db8708e`).
- Doctor notification via n8n (committed `1a9b500`).
- Feedback / retrain loop (this task; not yet committed).

## Next Files (optional / backlog)

- Encrypted gRPC transport.
- Risk monitoring on history.
- Cross-host deployment docs.
- Corpus expansion.

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

## Status

Feedback / retrain loop complete and verified end-to-end against the live
n8n + FastAPI (API restarted with `DATASET_DIR`; n8n restarted with
`N8N_BLOCK_ENV_ACCESS_IN_NODE=false` + `DOCTOR_NOTIFY_WEBHOOK`). Repo not
yet committed (user drives commits).