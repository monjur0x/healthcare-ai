# Current Context

## Current Milestone

Milestone 13 — Doctor-friendly Clinical Assessment page (paper §11/§12
dashboard UX) — implementation complete and verified by tests; not yet
committed. Working tree has the uncommitted changes (10 files) from this
task.

## Current Module

`frontend/streamlit_app.py` (assessment tab) · `frontend/dashboard/`
(`client.py`, `clinical.py`) · `backend/api/` (`routes.py`,
`services.py`, `schemas.py`)

## Current Task

Reworked the Clinical Assessment page into a model/preset-driven,
doctor-friendly form while keeping it a thin presentation layer (no ML
in the frontend, existing FastAPI + n8n + CrewAI architecture intact).

1. **Backend preset + CSV support**
   - `GET /api/v1/presets` — per-preset schemas (name, dataset, target,
     available, feature_names, classes) read from trained artifacts
     `artifacts_dir/<preset>/global_model.joblib`
   - `POST /api/v1/analyze/csv` — `AnalyzeCSVRequest` (base64-decoded
     `csv: bytes` + patient/markers/recommendations) →
     `CSVPipeline().run(csv)` → first-row prediction via the existing
     `analyze()` path
   - `ModelInfo` now carries `preset` (the served model's training
     preset), set by `AnalysisService.train()` and reported by
     `model_info()`
2. **Dashboard assessment tab** — Assessment Type selector (presets when
   available, else served model); Patient Context (name/id/age) separated
   from model features; model-driven "Clinical measurements" with human
   labels + verified units + integer formatting; blood-pressure entry
   now returns `float | None` (no silent 80.0 substitution); pre-run
   Assessment summary; validation before submit; train-on-demand when the
   selected preset differs from the served model (direct route → 
   `client.train`, n8n route → `preset`/`train` in the webhook payload);
   Input method toggle **Manual Entry / CSV Upload** (CSV routes directly
   to `/api/v1/analyze/csv`; n8n workflow does not carry files);
   disclaimer caption throughout.
3. **Tests** — frontend suite 54 passing (client +5, clinical +6,
   smoke +3 new: selector adapts form, train-on-demand direct route,
   CSV upload), backend suite 340 passing (+5 api tests). Lint clean
   (black/ruff from `frontend/` and `backend/`).

## Completed

- Milestones 1–12 — prior context (committed + pushed).
- Milestone 13 (this session, uncommitted):
  - `backend/api/schemas.py` — `PresetInfo`, `AnalyzeCSVRequest`,
    `ModelInfo.preset`
  - `backend/api/services.py` — `active_preset`, `presets_info()`,
    `analyze_csv()`, `model_info()` preset field
  - `backend/api/routes.py` — `GET /api/v1/presets`,
    `POST /api/v1/analyze/csv`
  - `frontend/dashboard/client.py` — `presets()`, `train(preset, ...)`,
    `analyze_csv()` (base64), n8n `preset`/`train` kwargs
  - `frontend/dashboard/clinical.py` — `DISPLAY_LABELS` (all four
    presets, no raw-name suffixes), `FEATURE_UNITS`, `INTEGER_FEATURES`,
    `feature_unit()`, `is_integer_feature()`, `validate_feature_values()`,
    `assessment_summary()`
  - `frontend/streamlit_app.py` — assessment tab rewrite + CSV upload
    mode, `CLINICAL_DISCLAIMER`, `fetch_presets_info()`,
    `assessment_type_label()`, `model_matches_preset()`, blood-pressure
    widget returning `float | None`
  - Tests: frontend 54 (+14), backend 340 (+5); black + ruff clean
- Backend full suite **340 passing**; frontend suite **54 passing**.

## Next Files (optional / backlog)

- Commit the uncommitted changes (user's call — no commit was made).
- Backlog candidates (pick one): patient persistence + history;
  mortality/readmission models (Results page still "not estimated");
  SHAP explainability; CSV path through the n8n end-to-end workflow;
  multimodal fusion so the summary no longer shows "image: Not provided";
  local/open-source LLM provider for the crew.

## Design Notes

- The dashboard remains a thin view layer (ADR-010): all prediction/RAG/
  crew reasoning stays server-side; the assessment tab only renders the
  schema the backend reports and delegates.
- One tabular model is served at a time. When the selected preset differs
  from the served model, the dashboard trains the preset on demand
  (direct route → `client.train`; n8n route → `preset` + `train` in the
  webhook payload, already supported by `healthcare-endtoend.json`).
- `presets_info()` derives schemas from trained artifacts only (datasets
  are not in the repo); a preset with no trained artifact is reported
  `available: false` and the tab asks the user to train it first.
- Uploaded CSV analysis re-fits the `CSVPipeline` scaler on the uploaded
  rows (fresh fit) — a documented limitation for inference-time
  consistency, same behaviour as the existing analyze path.
- Blood-pressure widget: `SYS/DIA` → diastolic (PIMA convention); invalid
  input returns `None` and blocks submission with a clear error (no
  silent `80` fallback anymore).
- Lint/tests: frontend from `frontend/`, backend from `backend/`; never
  ruff from the repo root.

## Status

Milestones 1–12 committed/pushed; Milestone 13 changes are **uncommitted**
(10 modified files). Verify with `git status`. Frontend `pytest -q` from
`frontend/` = 54 passed; backend `pytest -q` from `backend/` = 340
passed.