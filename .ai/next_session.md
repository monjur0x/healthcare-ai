# Next Session

## Objective

The doctor-friendly Clinical Assessment rework (Milestone 13) is
implemented and verified (frontend 54, backend 340, lint clean) but
**not yet committed** (10 modified files). The natural first step is to
review `git diff` and commit, then pick a backlog direction.

## Done This Session (uncommitted)

- Backend: `GET /api/v1/presets` (per-preset schemas from trained
  artifacts), `POST /api/v1/analyze/csv` (base64 CSV → CSVPipeline →
  first-row analyze), `ModelInfo.preset`, `active_preset` in
  `AnalysisService`.
- Frontend: Assessment Type selector (preset-driven, train-on-demand),
  Patient Context separated from model features, human labels + units +
  integer formatting, blood-pressure `float | None` (no silent fallback),
  pre-run summary + validation, disclaimer, and a **Manual Entry / CSV
  Upload** input toggle (CSV routes directly to FastAPI — n8n cannot
  carry files).
- Tests: frontend +14 (client presets/train/analyze_csv, clinical
  units/integers/validation/summary, smoke: selector adapts form,
  train-on-demand direct route, CSV upload), backend +5 (presets schema,
  analyze_csv success/422/503). Black + ruff clean.

## Optional Next Steps

1. Commit Milestone 13 (await user instruction): one or two focused
   commits (e.g. `feat(api)` presets + CSV analyze endpoint, then
   `feat(dashboard)` assessment tab rework).
2. Backlog candidates (pick a direction):
   - Patient persistence + history in the dashboard (each assessment is
     entered fresh)
   - Backend mortality/readmission risk models (Results page still shows
     "not estimated")
   - Model-derived / SHAP explainability for the decision report
   - Route CSV uploads through the n8n end-to-end workflow (currently
     direct-to-FastAPI only; the workflow's HTTP request node would need
     a multipart/base64 body)
   - Multimodal fusion so the assessment summary can show an image result
     instead of the static "Not provided"
   - Open-source / local LLM provider for the crew (`CrewSettings.
     LLM_PROVIDER` currently only has `google`)
3. Run tests + lint from the right directory: `frontend/` for the
   dashboard suite + ruff/black; `backend/` for the API suite. Never ruff
   from the repo root.

## Open Questions

- Whether to commit now (user decides; nothing was committed this
  session).
- `gemini-3.7-flash` still returns transient 503 "high demand";
  `gemini-3.6-flash` verified stable — consider switching
  `CrewSettings.LLM_MODEL` default.