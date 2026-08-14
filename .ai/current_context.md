# Current Context

## Current Milestone

Milestone 7 — Streamlit dashboard (complete)

## Current Module

frontend/

## Current Task

The Streamlit dashboard, its client, and tests are complete and verified
(end-to-end against a live backend). Not yet committed — awaiting user
go-ahead. Next: API hardening / flwr deployment path.

## Completed

- Milestone 1: preprocessing (CSV + image + multimodal), 70 tests
- Milestone 2: models, evaluation, federated tie-in, sync FedAvg server,
  CSV → FedAvg demo, CNN federation, federated metrics, image FedAvg
  demo
- Milestone 3: RAG module + demo (38 tests + 2 demo smoke tests)
- `backend/CrewAI/orchestrator/` — CrewAI multi-agent orchestration
  consuming preprocessing / models / rag:
  - `config.py` — `CrewSettings` (env prefix `CREW_`): optional LLM
    provider/model/key, crew verbosity/memory, RAG top-k, risk + marker
    thresholds
  - `exceptions.py` — `CrewError` + tool / report /
    `LLMNotConfiguredError`
  - `schemas.py` — `PatientInfo`, `PredictionResult`, `RiskResult`,
    `EvidenceItem`, `ClinicalReport` (pydantic, `to_dict()`)
  - `services.py` — deterministic LLM-free core: `run_prediction`,
    `assess_risk`, `retrieve_evidence`, `assemble_clinical_report`
  - `prompts.py` — seven agent profiles + task descriptions + report
    schema instructions
  - `tools.py` — crewai `BaseTool` wrappers: `PredictionTool`,
    `RiskAssessmentTool`, `RAGRetrievalTool`, `ClinicalReportTool`
  - `agents.py` / `tasks.py` — seven agents (Patient Analysis, Disease
    Prediction, Medical Research, Treatment, Explainability, Risk
    Monitoring, Report Writing) and chained tasks; LLM bound only on
    the LLM path (hermetic construction)
  - `crew.py` — `ClinicalCrew`: `run_analysis()` (offline deterministic
    pipeline, ADR-008), `run_llm()` (CrewAI kickoff when
    `CREW_LLM_API_KEY` set, deterministic fallback), `run()` selects by
    config
  - Tests: 25 passing (hermetic, no LLM keys)
- `examples/clinical_crew_demo.py` — CSV → `CSVPipeline` →
  `TabularClassifier` → `ClinicalCrew.run_analysis()` → clinical report
  (`report.json`); verified on diabetes; built-in or `--corpus-dir`
  knowledge base; smoke tests (2 passing)
- Cleanup: removed the superseded old CrewAI demo (`app/`,
  `tests/test_healthcare.py`, `Dockerfile`, `docker-compose.yml`,
  `requirements.txt`, `.env.example`, `README.md`) and untracked its
  generated `artifacts/` (`.gitignore` now covers `artifacts/`); the old
  demo's DP module is recorded in the backlog for a future port
- `backend/api/` — FastAPI module (Milestone 5, complete):
  - `config.py` — `APISettings` (env prefix `API_`): server metadata,
    `MODEL_PATH`, `CORPUS_DIR`, optional `API_TOKEN`, CORS origins
  - `exceptions.py` — `APIError` + `ServiceUnavailableError` /
    `InvalidInputError` / `AuthenticationError` / `NotFoundError`
  - `schemas.py` — `PredictRequest`, `RetrieveRequest`,
    `AnalyzeRequest`, `HealthResponse`; responses reuse orchestrator
    schemas (`PredictionResult` / `EvidenceItem` / `ClinicalReport`)
  - `services.py` — `AnalysisService` facade (lazy model load + RAG
    corpus ingest + deterministic crew analysis) translating domain
    exceptions into typed `APIError`s; `load_predictive_model`,
    `build_rag_pipeline` (corpus dir or built-in `DEFAULT_CORPUS`)
  - `routes.py` — `/api/v1/predict`, `/api/v1/retrieve`,
    `/api/v1/analyze` (validation + delegation only; optional
    bearer-token auth via router dependency)
  - `main.py` — `create_app()` factory (DI service via `app.state`,
    CORS, `APIError` → JSON handler) + module-level `app` for uvicorn
  - Tests: 19 passing (`test_services.py` real fitted model,
    `test_api.py` hermetic fake service); `fastapi` / `uvicorn[standard]`
    added to `backend/requirements.txt`
- Full suite **241 passing** (`pytest preprocessing/tests models/tests evaluation/tests federated/tests rag/tests examples/tests CrewAI/orchestrator/tests api/tests`)
  — black / isort / ruff clean
- `n8n/` orchestration (Milestone 6, complete — uncommitted):
  - `clinical-analysis.json` — webhook `/webhook/healthcare-analyze` →
    `POST /api/v1/analyze` → validate + summarize → structured
    `status: success|error` response (error branch merges HTTP and
    validation failures)
  - `clinical-pipeline-modality.json` — webhook
    `/webhook/healthcare-pipeline` → normalize → switch on `modality`
    (image / csv) → `/api/v1/analyze` with matching `input_type` →
    merged summary or merged errors
  - Optional bearer-token auth via an `httpHeaderAuth` credential
    (placeholder reference; fine while `API_TOKEN` unset)
  - `README.md` — import, config, payloads, smoke test
  - Removed stale `workflow.json` / `workflow2.json` (they targeted the
    removed old demo endpoints)
- `frontend/` Streamlit dashboard (Milestone 7, complete — uncommitted):
  - `dashboard/client.py` — `HealthcareAPIClient` (httpx): `health` /
    `predict` / `retrieve` / `analyze`; optional bearer token; typed
    `HealthcareAPIError` (status + code) from backend error detail
  - `streamlit_app.py` — thin view layer: sidebar backend URL/token +
    live health indicator; tabs Clinical Analysis / Prediction /
    Evidence Retrieval / Info (report render, probability bar chart,
    evidence score bars, JSON download)
  - `dashboard/tests/` — 7 client tests (`httpx.MockTransport`) + 2
    AppTest smoke tests; `frontend/requirements.txt`
  - Verified live: health, analyze (3 evidence items), retrieve,
    503 predict without configured model
- Full suite **241 backend + 9 frontend passing** — black / isort /
  ruff clean (frontend linted from `frontend/`)

## Next Files (backend)

- `federated/` — real flwr `run_simulation` / networked `ServerApp`
  (blocked: `ray` not installed); privacy budget metrics
- Orchestrator LLM path: wire a provider (needs `crewai[google-genai]`
  or similar + API key; never commit secrets)
- API hardening: full OAuth, file upload endpoint, deployment container,
  downstream n8n storage/notification branches

## Design Notes

- ADR-008: the crew runs a deterministic tool pipeline by default
  (prediction → risk → evidence → report) that needs no LLM and is
  fully testable; the CrewAI layer is optional narrative enrichment.
- ADR-009: FastAPI routes delegate to `AnalysisService` (no business
  logic in routes); domain exceptions are translated to typed
  `APIError`s (status + code) at the service boundary; optional static
  bearer-token auth via `API_TOKEN`; service injected through
  `app.state` for hermetic route tests.
- Agents/tasks/crew construct without an LLM key; `create_agents(llm=...)`
  binds the provider only on the LLM path. CrewAI 1.15 warns internally
  (deprecations) — unrelated to this module.
- `ClinicalCrew` needs `features` (full preprocessed row) when `model`
  is set; `markers` (raw clinical values) feed risk factor flags.
- CrewAI venv (`backend/CrewAI/.venv-opencode`) has crewai 1.15.11,
  pydantic 2.12, qdrant-client, sentence-transformers, flwr, torch,
  fastapi 0.138, uvicorn, httpx, streamlit 1.61.
- ADR-010: the frontend is a Streamlit dashboard that is a thin client
  over the FastAPI backend (no reasoning client-side); a future Next.js
  dashboard can reuse the same endpoints.
- The old `CrewAI/app/*` demo was removed (superseded by
  `preprocessing/`, `models/`, `federated/`, `rag/`, and
  `CrewAI/orchestrator/`); the production CrewAI module is
  `CrewAI/orchestrator/`.

## Status

Milestone 7 (Streamlit dashboard) complete and verified end-to-end; the
frontend work is uncommitted, awaiting user go-ahead to commit + push.
Next milestone is the real flwr deployment path, then API hardening.