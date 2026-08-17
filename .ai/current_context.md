# Current Context

## Current Milestone

Milestone 13.1 — two production bug fixes, implemented and verified by
tests + live API checks; **not yet committed** (working tree has the
uncommitted changes).

## Current Module

`backend/preprocessing/csv/scaler.py` · `backend/models/csv/tabular.py` ·
`backend/CrewAI/orchestrator/services.py` · `backend/api/services.py`

## Current Task

1. **Inference-time scaling fix** — manual / n8n structured-feature input
   was fed raw (unscaled) values into a model trained on scaled features,
   so the Disease Risk Score always saturated to 1.00 / HIGH / 100%
   (it was a real bug, not a placeholder). The model artifact now persists
   the training scaler params (`CSVScaler.params()` / `from_params()`,
   mean/std in `ScalingReport`, `TabularClassifier.set_scaler_params` +
   joblib round-trip), `run_prediction` applies them to raw input unless
   `preprocessed=True`, and the flag is threaded through `ClinicalCrew` /
   `AnalysisService.analyze` / `analyze_csv` / the baseline study so
   already-scaled inputs are never double-scaled.
2. **CrewAI LLM wiring** — `AnalysisService.analyze` hardcoded the
   LLM-free `crew.run_analysis()`; it now defaults `use_llm=True` and
   calls `crew.run()` (LLM orchestration when `CREW_LLM_API_KEY` is set,
   deterministic fallback otherwise). Baseline study forces
   `use_llm=False`.

## Completed

- Milestones 1–13 + CSV-through-n8n follow-up — committed + pushed.
- Milestone 13.1 (this session, uncommitted):
  - `preprocessing/csv/scaler.py` — `ScalingReport` mean/std,
    `params()` / `from_params()`
  - `models/csv/tabular.py` — `scaler_params` persistence (save/load)
  - `CrewAI/orchestrator/services.py` — `run_prediction` applies the
    persisted scaler; `preprocessed` flag
  - `CrewAI/orchestrator/crew.py` — `preprocessed` threaded into
    `run_analysis`
  - `api/services.py` — `prepare_tabular_data` returns
    `(features, labels, scaler_params)`; `train()` stores them; `analyze`
    gains `use_llm=True` (crew.run) + `preprocessed`; `analyze_csv`
    passes `preprocessed=True` and reuses the persisted scaler
  - `scripts/baseline_study.py` — `preprocessed=True`, `use_llm=False`
  - Tests: +2 in `test_prediction_service.py` (scaling equivalence +
    preprocessed skip); `test_services.py` unpacks the 3-tuple
  - Retrained `backend/artifacts/diabetes/global_model.joblib` with
    scaler params; backend restarted (port 8000)
- Backend core suite **223 passing**; black/ruff clean on touched files.
- Live-verified: reference patient (6/148/72/35/0/33.6/0.627/50) manual
  entry now → prediction `0` @ **62.8%** confidence (risk 0.63 / high)
  instead of the saturated 100%; CSV path ~71% (unchanged, no
  double-scaling).

## Next Files (optional / backlog)

- Commit the uncommitted changes (user's call).
- Backlog candidates (pick one): fix 4 pre-existing frontend smoke-test
  failures; patient persistence + history; mortality/readmission models;
  SHAP explainability; encoder (categorical level map) persistence;
  multimodal fusion; local/open-source LLM provider for the crew.

## Design Notes

- Scaling belongs at the inference boundary (`run_prediction`), not in
  the pipeline re-fit: the CSV path preprocesses once with the persisted
  scaler and marks its features `preprocessed=True`; raw manual/n8n input
  flows through unscaled and gets scaled exactly once.
- `use_llm` defaults to `True` for `analyze` (matches `analyze_image`);
  `crew.run()` already falls back to the deterministic path when no
  `CREW_LLM_API_KEY` is configured, so the default is safe offline.
- The 4 frontend smoke failures are pre-existing on `main` (verified by
  stashing the working tree) — recorded in BACKLOG, not fixed here.
- Lint/tests: backend from `backend/`; frontend from `frontend/`; never
  ruff from the repo root.

## Status

Milestones 1–13 committed/pushed. Milestone 13.1 is **uncommitted**
(10 modified files). Backend `pytest -q` (core suite) = 223 passed;
frontend = 58 passed with 4 pre-existing smoke failures.