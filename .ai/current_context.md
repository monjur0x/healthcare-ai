# Current Context

## Current Milestone

Milestone 13.1 — two production bug fixes plus the "CrewAI is not
optional" hardening, committed (`3bd95e8`, **not pushed**) and verified
by tests + live API checks.

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
2. **CrewAI is mandatory, not optional** — `AnalysisService.analyze` and
   `analyze_image` no longer expose a `use_llm` opt-out: every analysis
   goes through `crew.run()` (CrewAI agentic path when an LLM is
   configured, deterministic fallback only when the LLM is unavailable or
   its output cannot be parsed). The baseline study and the demo script no
   longer bypass the crew; the baseline study calls
   `ClinicalCrew.run_analysis()` directly for reproducibility.
3. **NVIDIA NIM provider** — `CrewSettings.LLM_BASE_URL` enables a custom
   OpenAI-compatible endpoint; `backend/.env` is switched to NVIDIA
   (`nvidia/nemotron-3.5-lightning-30b-a3b` @
   `https://integrate.api.nvidia.com/v1`) because Gemini's free tier was
   rate-limited (5 req/min) and `meta/llama-3.3-70b-instruct` hung on
   NVIDIA. Live-verified: full 7-agent kickoff completes (~6 min) and the
   merged report keeps the deterministic prediction `0` @ 71.8% / risk
   `high` / 3 evidence items while the LLM enriches the narrative.

## Completed

- Milestones 1–13 + CSV-through-n8n follow-up — committed + pushed.
- Milestone 13.1 (this session):
  - `preprocessing/csv/scaler.py` — `ScalingReport` mean/std,
    `params()` / `from_params()`
  - `models/csv/tabular.py` — `scaler_params` persistence (save/load)
  - `CrewAI/orchestrator/services.py` — `run_prediction` applies the
    persisted scaler; `preprocessed` flag
  - `CrewAI/orchestrator/crew.py` — `preprocessed` threaded into
    `run_analysis`
  - `api/services.py` — `prepare_tabular_data` returns
    `(features, labels, scaler_params)`; `train()` stores them; `analyze`
    and `analyze_image` always call `crew.run()` (no `use_llm` flag);
    `analyze_csv` passes `preprocessed=True` and reuses the persisted
    scaler
  - `scripts/baseline_study.py` — `preprocessed=True`; builds
    `ClinicalCrew` directly and calls `run_analysis()` for
    reproducibility (no `use_llm` param)
  - `examples/clinical_crew_demo.py` — now uses `crew.run()` (prefers
    LLM when configured) instead of forcing the offline path
  - `CrewAI/orchestrator/config.py` — `LLM_BASE_URL` for custom
    OpenAI-compatible endpoints (NVIDIA NIM)
  - `CrewAI/orchestrator/agents.py` — `_agent_llm()` returns a CrewAI
    config dict (`custom_openai=True`) when `LLM_BASE_URL` is set
  - `CrewAI/orchestrator/crew.py` — `run_llm` skips the Gemini env-var
    setup for custom endpoints and merges the LLM report over the
    deterministic base (`_merge_llm_over_base`) so prediction/risk/
    evidence are never dropped
  - `backend/conftest.py` — autouse fixture forces `LLM_API_KEY=""` so
    tests never hit the LLM (hermetic, deterministic)
  - Tests: +2 in `test_prediction_service.py` (scaling equivalence +
    preprocessed skip); `test_services.py` unpacks the 3-tuple; +3 in
    `test_agents.py` (NVIDIA config); +2 in `test_crew.py` (merge)
  - Retrained `backend/artifacts/diabetes/global_model.joblib` with
    scaler params; backend restarted (port 8000)
- Commit `3bd95e8` `fix(api): apply persisted scaler at inference and
  wire crew LLM path` — 16 files, +427/−112, working tree clean; **not
  pushed**.
- Backend full suite **342 passing**; black/ruff clean on touched files.
- Live-verified with NVIDIA: `/api/v1/analyze` runs the full 7-agent
  kickoff (~6 min) and returns the merged report — prediction `0` @
  71.8% confidence (risk high), 3 evidence items, LLM-enriched summary.
  Gemini's free tier was rate-limited (5 req/min, 429/503), so the
  project now uses NVIDIA NIM.

## Next Files (optional / backlog)

- Push commit `3bd95e8` (user's call).
- Backlog candidates (pick one): fix 4 pre-existing frontend smoke-test
  failures; patient persistence + history; mortality/readmission models;
  SHAP explainability; encoder (categorical level map) persistence;
  multimodal fusion; local/open-source LLM provider for the crew (the
  free-tier Gemini quota of 5 req/min is the current bottleneck).

## Design Notes

- Scaling belongs at the inference boundary (`run_prediction`), not in
  the pipeline re-fit: the CSV path preprocesses once with the persisted
  scaler and marks its features `preprocessed=True`; raw manual/n8n input
  flows through unscaled and gets scaled exactly once.
- CrewAI is always invoked: no `use_llm` flag exists on the service
  methods anymore. `crew.run()` prefers the LLM path when an LLM is
  configured and falls back to the deterministic pipeline only as a
  safety net when the kickoff fails or its output cannot be parsed
  (ADR-008 still documents the reproducible path for research).
- The LLM enriches the narrative, never the numbers: `_merge_llm_over_base`
  keeps the deterministic prediction/risk/evidence and overlays the LLM's
  summary, context, recommendations, and notices.
- LLM provider is configurable via `.env`: native `provider/model` string
  when `CREW_LLM_BASE_URL` is empty (Gemini), or a custom OpenAI-
  compatible endpoint when set (NVIDIA NIM). NVIDIA is currently active.
- The baseline study and its tests use `ClinicalCrew.run_analysis()`
  directly — research reproducibility must never depend on an external LLM.
- Tests are hermetic: `backend/conftest.py` clears `LLM_API_KEY` for the
  whole suite so no test makes a network call.
- The 4 frontend smoke failures are pre-existing on `main` (verified by
  stashing the working tree) — recorded in BACKLOG, not fixed here.
- Lint/tests: backend from `backend/`; frontend from `frontend/`; never
  ruff from the repo root.

## Status

Milestones 1–13 committed/pushed. Milestone 13.1 + NVIDIA provider work
committed locally as `3bd95e8` (base, **not pushed**) with additional
uncommitted changes (NVIDIA config, merge fix, mandatory crew). Backend
`pytest -q` (core suite) = 111 passed in the touched dirs (342 full);
frontend = 58 passed with 4 pre-existing smoke failures.