# Current Context

## Current Milestone

Milestone 11 — doctor-facing CDS dashboard (Milestone 7 rework) (complete, uncommitted)

## Current Module

frontend/ (streamlit_app.py · dashboard/clinical.py · dashboard/client.py · tests) · n8n/ · docs/ · .ai/

## Current Task

Reworked the research-facing Streamlit dashboard into a doctor-friendly
Clinical Decision Support interface aligned with the research workflow
(Patient Data → Federated Prediction → Disease Prediction Agent → RAG →
Treatment Agent → Explainability → n8n → Doctor Dashboard). No backend
changes were made.

1. `frontend/dashboard/clinical.py` (NEW) — pure presentation helpers:
   `group_features` (Vital Signs / Clinical Measurements / Medical
   History / Additional Model Features), `feature_label`, `feature_bounds`,
   `is_flag_feature`, `build_analyze_payload`, `analysis_stages`,
   `explanation_sections`, `output_availability`.
2. `frontend/dashboard/client.py` — added `analyze_via_n8n()` (POSTs to
   the `/webhook/healthcare-endtoend` webhook, reads the full
   `ClinicalReport` from `body["report"]`), `n8n_health()` (probe
   `{n8n}/healthz`), shared `_analyze_payload()`.
3. `frontend/streamlit_app.py` — five-tab CDS layout: Overview, Clinical
   Assessment (model-driven grouped form, one **Analyze Patient** action),
   Imaging (upload → preview → analyze, graceful when no image model),
   Results (six research outputs, honest "not estimated" for mortality /
   readmission risk), System Status (live FastAPI / ML / RAG / CrewAI /
   n8n probes + effective route). Sidebar: backend URL, optional API
   token, n8n URL, Advanced → route radio (Automatic / Via n8n / Direct).
4. `n8n/healthcare-endtoend.json` — Code node now returns `report` in the
   webhook response (workflow JSON validated).
5. `frontend/pyproject.toml` (NEW) — mirrors `backend/pyproject.toml`
   tooling config (black 88 / isort / ruff E,F,W,I,B,UP,SIM,RUF,PIE,BLE,A)
   so the frontend lints standalone (ruff config resolves from cwd, but
   black/isort resolve from file location).

Frontend suite **35 passing** (+22), backend **326 passing** (unchanged),
lint clean (black / ruff / isort, frontend from `frontend/`, backend from
`backend/`). Live-verified against a running backend: `/api/v1/model`
(image-only brain CNN), `analyze` with no tabular model → evidence
without prediction (graceful), real glioma scan via `analyze_image` →
prediction (meningioma @ 69%) + risk + evidence through the dashboard
client. The n8n webhook path was NOT live-tested (no local n8n
instance) — covered by hermetic unit tests + JSON validation.

## Completed

- Milestones 1–10 — see prior context (all committed and pushed).
- Milestone 11 (this session, uncommitted):
  - `frontend/dashboard/clinical.py` — grouped feature helpers,
    post-hoc pipeline stages, derived explainable decision report
  - `frontend/dashboard/client.py` — `analyze_via_n8n`,
    `n8n_health`, shared `_analyze_payload`
  - `frontend/streamlit_app.py` — 5-tab CDS dashboard
  - `n8n/healthcare-endtoend.json` — Code node returns full `report`
  - `frontend/pyproject.toml` — standalone frontend tooling config
  - Tests: `tests/test_clinical.py` (14), client n8n (6),
    `test_app_smoke.py` rewritten (7); frontend suite 35
  - Live validation: analyze (no tabular model) + analyze_image
    (brain MRI) via the dashboard client against a real backend

## Next Files (frontend / downstream)

- Live n8n end-to-end verification from the dashboard once an n8n
  instance is running (activate `healthcare-endtoend.json`)
- Persistent patient records + history in the dashboard
- Backend mortality-risk / readmission-risk models so Results can stop
  showing "not estimated"
- Backend SHAP / feature-importance explainability for the Explainable
  Decision Report
- Uncommitted changes: review `git diff`, then commit as focused
  `feat(frontend)` commits and push

## Design Notes

- ADR-010 (thin view layer): the dashboard only collects inputs, calls
  the backend (direct or via n8n), and renders. No reasoning in the
  frontend; `clinical.py` helpers are pure and unit-tested.
- n8n stays core: `Automatic` route uses n8n when `healthz` is
  reachable, else falls back to FastAPI; `N8N_ENABLED=0` documented as
  the dev-only direct route.
- Honest states: mortality risk, readmission risk, missing treatment
  recommendations, missing evidence, and no-image-model are rendered
  explicitly rather than fabricated. Pipeline stages are derived
  post-hoc from report fields (no fake progress).
- Evidence is rendered without raw vector ids / similarity scores;
  `document_id` / `score` stay in the report JSON for research use.
- Lint/tests: from `frontend/` — `ruff check streamlit_app.py
  dashboard/`, `black --check`, `isort --check-only`,
  `pytest dashboard/tests -q` (CrewAI venv, streamlit installed there).
  Backend lint from `backend/`. Do NOT run ruff from the repo root
  (frontend now has its own config; backend config resolves from
  `backend/`).
- Two identical auto-generated widget IDs caused
  `StreamlitDuplicateElementId` when the report rendered on both the
  Assessment and Results tabs in one run — fixed with per-call-site
  `download_key` values.
- Black does not split long lambdas — the smoke tests use module-level
  `fake_analyze` functions instead of inline lambdas.

## Status

Milestones 1–10 committed/pushed; Milestone 11 (CDS dashboard rework)
implemented, tested, lint-clean, and live-validated against a running
backend, but NOT yet committed. Frontend 35 passing, backend 326
passing. Next: commit Milestone 11, then pick from the deferred items
(n8n live verification, patient persistence, mortality/readmission
risk, SHAP explainability).