# Next Session

## Objective

Continue after Milestone 11 (doctor-facing CDS dashboard rework, implemented
but **uncommitted**). First task: review `git diff`, commit as focused
commits, and push. Then pick from the deferred items below. Read
`docs/DEVELOPMENT_STATUS.md` + `.ai/current_context.md` first (AGENTS.md
workflow).

## Suggested Steps

1. Read `docs/SYSTEM_SPECIFICATION.md`, `docs/SOFTWARE_ARCHITECTURE.md`,
   `docs/DEVELOPMENT_STATUS.md`, `docs/BACKLOG.md`, `docs/DECISIONS.md`,
   `.ai/current_context.md`.
2. Commit Milestone 11 (the working tree is dirty):
   - `feat(frontend): add doctor-facing CDS dashboard (5-tab)`
     (`frontend/streamlit_app.py`, `frontend/dashboard/clinical.py`)
   - `feat(frontend): route analyses through the n8n end-to-end webhook`
     (`frontend/dashboard/client.py`)
   - `feat(n8n): return full clinical report from end-to-end webhook`
     (`n8n/healthcare-endtoend.json`)
   - `test(frontend): clinical helpers + n8n client + app smoke`
     (tests) and `chore(frontend): standalone tooling config`
     (`frontend/pyproject.toml`) + docs
3. **Live n8n verification**: start n8n (`scripts/run_system.sh start`),
   import + activate `n8n/healthcare-endtoend.json`, then drive the
   dashboard's `Via n8n workflow` route end-to-end (the Milestone 11
   session only validated the direct path live; the n8n path is covered
   by hermetic tests only).
4. Pick one backlog direction:
   - **Patient persistence**: add a lightweight patient/history store so
     the dashboard can list past assessments instead of "each assessment
     is entered fresh" (currently documented as future work).
   - **Mortality / readmission risk**: extend `ClinicalReport` +
     `assess_risk` with these outputs so the Results page can replace
     the "not estimated" placeholders with real data.
   - **Explainability**: add model-derived feature importance /
     SHAP-style attribution for the Explainable Decision Report (now
     derived from prediction / risk outputs only).
   - **Backend**: OAuth, file-upload endpoint, deployment container,
     RAGAS / agent-metrics API exposure, production DP pass
     (`secure_mode=True`), real flwr `run_simulation` (blocked: `ray`).
5. Add unit tests under the module's `tests/` directory; keep test-file
   basenames unique across `backend/`.
6. Run tests and lint:
   - Frontend (from `frontend/`): `pytest dashboard/tests -q`; `ruff
     check streamlit_app.py dashboard/`; `black --check`; `isort
     --check-only` (CrewAI venv binaries).
   - Backend (from `backend/`): `pytest preprocessing/tests
     models/tests evaluation/tests federated/tests rag/tests
     examples/tests CrewAI/orchestrator/tests api/tests -q`; `ruff
     check . ../frontend`.
   - Never run ruff from the repo root; the frontend now has its own
     `frontend/pyproject.toml` (black/isort resolve config from file
     location, ruff from cwd).
7. Update `docs/DEVELOPMENT_STATUS.md`, `docs/CHANGELOG.md`,
   `docs/BACKLOG.md`, `docs/DECISIONS.md` (if a new ADR),
   `.ai/current_context.md`, `.ai/next_session.md`.

## Conventions Reminder

- Preprocessing stays in `preprocessing/`, models in `models/`,
  retrieval in `rag/`, orchestration in `CrewAI/orchestrator/`,
  API in `api/`, orchestration only in `n8n/`, view layer only in
  `frontend/` (ADR-010). Pure presentation helpers go in
  `frontend/dashboard/clinical.py` (no Streamlit in the helper module);
  widgets stay in `streamlit_app.py`.
- The dashboard routes analyses through n8n (`analyze_via_n8n` →
  `/webhook/healthcare-endtoend`, full report read from `body["report"]`)
  with an Automatic / Via-n8n / Direct fallback; `N8N_ENABLED=0` is the
  dev-only direct route. The workflow's Code node returns `report`.
- Unsupported outputs (mortality / readmission risk, empty treatment
  recommendations, no image model) render honestly — never fabricate.
  Pipeline stages derive from real report fields; no fake progress.
- Report rendering happens on more than one page in the same run
  (Assessment + Results) — every Streamlit widget needs a unique
  `key`; the download button uses per-call-site `download_key`.
- Streamlit AppTest: black does not split long lambdas — define small
  `fake_*` functions in tests; `file_uploader.set_value` needs a
  `(name, bytes, mime)` tuple; a real (e.g. PIL-generated) image is
  required for `st.image`.
- Log via `get_logger(__name__)`; raise module exceptions from the
  module's `exceptions.py`.
- Background servers for live checks: start via
  `scripts/run_system.sh start` (`N8N_ENABLED=0` to skip n8n). Avoid
  `pkill -f "uvicorn api.main"` self-matches — use the `[u]vicorn`
  bracket trick. Kill stale uvicorn PIDs before starting if
  `/api/v1/model` 404s (an old process serves old code).

## Open Questions

- Whether to commit Milestone 11 as one feature commit or split into
  focused `feat`/`test`/`docs` commits (AGENTS.md prefers one logical
  feature per commit).
- Whether mortality / readmission risk and SHAP explainability belong in
  the backend report schema now or are paper-evaluation-only.
- Whether the real `run_simulation` integration check belongs in the
  unit suite (Ray spawn is slow/flaky) or as a separate smoke script.
- Whether to keep `gemini-3.7-flash` as the CrewAI default (transient
  503 "high demand" on 2026-08-13) or fall back to `gemini-3.6-flash`,
  which verified working end-to-end.
- Whether a Next.js dashboard is still wanted on top of the Streamlit
  one (ADR-010 picks Streamlit for now).
- `PRESETS` duplication: the registry lives in `api/services.py`, but
  `examples/fedavg_demo.py` and `examples/clinical_crew_demo.py` keep
  their own copies — consider a shared registry if another consumer
  appears.