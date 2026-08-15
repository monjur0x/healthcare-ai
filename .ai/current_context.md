# Current Context

## Current Milestone

Milestone 9 — privacy-preserving federated learning (paper §8) (complete)

## Current Module

federated/ · models/csv/ · api/ · frontend/ · docs/ · .ai/

## Current Task

Ported the old demo's privacy layer into `backend/federated/privacy.py`
(Opacus DP-SGD + epsilon audit, pairwise-OTP `SecureAggregator`,
anonymization / pseudonymization, MIA-AUROC + leakage-rate metrics) and
wired it into `FederatedClient` / `FedAvgServer` / `FederatedMetrics` and
the API (`POST /api/v1/train` → `federated_metrics.privacy`). Live
verified on the diabetes preset (ε ≈ 2.14, 53.5% of budget, MIA-AUROC ≈
0.50, attack-resistance ≈ 1.0, leakage 0.0, DP + SecAgg). All privacy +
Milestone 8.1 + LLM/.env work uncommitted — awaiting user go-ahead to
commit + push.

Open question for the user: keep `gemini-3.7-flash` as the CrewAI default
(currently transient 503 "high demand") or switch to `gemini-3.6-flash`.

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
- Milestone 8 — functional end-to-end system (complete, committed:
  `1c1ec42` feat(api) train endpoint, `daeed76` feat(n8n) single
  workflow, `26911d9` chore(scripts) runner, `3fb5dcc` docs + ADR-011):
  - `api/` `POST /api/v1/train`: `TrainRequest`/`TrainResponse`
    schemas, `prepare_tabular_data` + `PRESETS` + `TrainResult` +
    `AnalysisService.train` in services (central fit default, federated
    FedAvg path with `federated:true`; fitted model replaces the service
    model so predict/analyze serve it immediately); `APISettings`
    gains `ARTIFACTS_DIR` / `DATASET_DIR`; +14 tests (API suite 33;
    full backend 255). Federated path supports `model_name='mlp'` only.
  - `n8n/healthcare-endtoend.json` — single workflow: webhook →
    optional train → analyze → write report JSON to disk
    (`/tmp/healthcare_reports/`, override `output_dir`) → structured
    success/error; removed `clinical-pipeline-modality.json`;
    `clinical-analysis.json` kept as minimal reference; README updated.
  - `scripts/run_system.sh` — one-command start/status/stop (trains
    default model via API, starts backend + dashboard, starts n8n docker;
    `N8N_ENABLED=0` skips n8n)
  - Root `README.md` — CPU-only step-by-step run guide
  - Live E2E verified on diabetes: train central (acc ~0.81) +
    federated (2 clients/2 rounds, full federated_metrics), predict,
    retrieve, analyze (prediction + high risk + 3 evidence items);
    dashboard + n8n (docker, editor+healthz 200) running
- Milestone 8.1 — image analysis + friendly dashboard input (complete,
  uncommitted — awaiting user go-ahead):
  - `models/image/cnn.py` — `ImageClassifier` now accepts string class
    labels (class order from `np.unique` instead of `int(label)`), so
    the brain-tumor folder classes work directly
  - `CrewAI/orchestrator/services.py` — `run_image_prediction(image_model,
    image)` builds a `PredictionResult` from a `(H,W,C)` array
    (`"image-cnn"`); `crew.py` — `ClinicalCrew(image_model=..., image=...)`
    branches `run_analysis` to the image path (raises
    `OrchestrationError` if image missing)
  - `api/` — `APISettings.IMAGE_MODEL_PATH` (env `API_IMAGE_MODEL_PATH`);
    `AnalyzeImageRequest` (base64 image decoded by Pydantic field
    validator — gotcha: `bytes` alone stores the base64 UTF-8 text, so
    an explicit `b64decode(validate=True)` validator is required);
    `ModelInfo` schema; `load_image_model()` + `AnalysisService.image_model`
    + `model_info()` + `analyze_image()`; routes `GET /api/v1/model`
    and `POST /api/v1/analyze/image`
  - `scripts/train_image_model.py` — trains the brain-tumor CNN
    (folder-per-class dataset, hold-out metrics, artifact → 
    `backend/artifacts/brain/global_model.pt`); trained at 64×64, 6
    epochs, batch 32, full training split (4480 imgs) → hold-out
    accuracy ~0.71, ROC-AUC ~0.92, F1 ~0.71
  - `scripts/run_system.sh` — `IMAGE_MODEL_PATH` default →
    `API_IMAGE_MODEL_PATH` to uvicorn; also fixed the `train_default_model`
    inline-Python indentation bug (now a proper multi-line `python -c`)
  - `frontend/dashboard/client.py` — `model_info()` + `analyze_image()`
    (base64 image, optional markers/recommendations)
  - `frontend/streamlit_app.py` — friendly per-feature numeric inputs
    (from `/api/v1/model` feature_names; raw JSON fallback when no
    model), patient form, marker inputs, Image (MRI upload) mode with
    preview, model metadata in Info tab
  - `backend/.env.example` — documents all env vars (API_ / CREW_ /
    MODEL_ / RAG_) incl. `CREW_LLM_API_KEY`; default CrewAI LLM model
    is now `gemini-3.7-flash`; README has an "Enabling the CrewAI LLM
    agents (Gemini)" section
  - LLM path made live-verifiable: all five settings classes
    `extra="ignore"` (shared `.env` safe); `run_llm` exports
    `CREW_LLM_API_KEY` → `GEMINI_API_KEY` (crewai 1.15 Gemini provider
    reads that env var); `crewai[google-genai]` installed. Verified:
    7 agents / 7 tasks / crew construct; `gemini-3.6-flash` accepts
    calls; `gemini-3.7-flash` returned transient 503 "high demand"
    (launched 2026-08-13); LLM parse-failure falls back to the
    deterministic report (by design)
  - Tests: backend 271 (+16), frontend 13 (+4); lint clean everywhere
- Full suite **271 backend + 13 frontend passing** — black / isort /
  ruff clean
- Milestone 9 — privacy-preserving federated learning (paper §8)
  (complete, uncommitted — awaiting user go-ahead):
  - `federated/privacy.py` — `PrivacyConfig`, `anonymize_frame`
    (`PII_PATTERNS`), `pseudonymize`, `train_with_differential_privacy`
    (Opacus DP-SGD + `get_epsilon` audit), `SecureAggregator` (pairwise
    one-time-pad adapted to `list[np.ndarray]`; masks only cancel under
    equal weights), `membership_inference_auroc` (ascending-rank sort so
    the orientation is correct), `data_leakage_rate`,
    `privacy_metrics_summary` (epsilon, budget-used %, MIA-AUROC,
    attack-resistance score clamped [0,1], leakage, mechanism). Ported
    from the removed old demo's `CrewAI/app/federated/privacy.py`
  - `models/csv/TorchMLPClassifier` — torch ReLU MLP on the same
    `get_parameters` / `set_parameters` / `partial_fit` contract as
    `TabularClassifier` (Opacus needs a `torch.nn.Module`); joblib
    payload `kind="torch_mlp"`; `module` property; exported from
    `models/csv/__init__.py` + `models/__init__.py`
  - `federated/client.py` — `FederatedClient` accepts `PrivacyConfig`;
    DP local training reports per-round epsilon in fit metrics
  - `federated/server.py` — `FedAvgServer(secure_aggregation=True)`
    aggregates via `SecureAggregator`; collects epsilons across rounds;
    `FederatedMetrics` gains `secure_aggregation`, `differential_privacy`,
    `epsilon` (worst-case per-client)
  - `api/` — `TrainRequest` gains `differential_privacy`,
    `noise_multiplier`, `max_grad_norm`, `privacy_delta`,
    `secure_aggregation`; `_train_federated` builds torch MLPs per client
    when DP is on, runs the MIA audit (members = client shards,
    non-members = hold-out), and returns `federated_metrics.privacy`
  - Dependency: `opacus` 1.6.0 installed in the backend venv
  - Live (API): diabetes preset, federated DP + SecAgg → accuracy ~0.72,
    ε ≈ 2.14 (53.5% of ε=4 budget), MIA-AUROC ≈ 0.5003,
    attack-resistance ≈ 0.9995, leakage 0.0, mechanism "DP-SGD (Opacus)
    + Secure Aggregation (pairwise OTP)"
  - Tests: backend 289 (+18), frontend 13; lint clean (black / isort /
    ruff) — fixed 2 unused-import / line-length / clamp issues found by
    ruff along the way
- Full suite **289 backend + 13 frontend passing** — black / isort /
  ruff clean

## Next Files (backend)

- Real flwr `run_simulation` / networked `ServerApp` (blocked: `ray`
  not installed)
- Production DP pass: re-run with Opacus `secure_mode=True` (currently
  `secure_mode=False` for experimentation speed — UserWarning asks for a
  final secure retrain before release)
- Orchestrator LLM path: settle `gemini-3.7-flash` (503 overloaded) vs
  `gemini-3.6-flash` default; wire a provider (needs `crewai[google-genai]`
  or similar + API key; never commit secrets)
- API hardening: full OAuth, file-upload endpoint for CSV / image,
  deployment container, downstream n8n storage/notification branches;
  split/online retraining UI for the image model
- Dashboard: picklists for categorical features, per-class image
  confidence histogram, latency metrics, privacy-metrics panel from
  `federated_metrics.privacy`

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
- ADR-011: training is an API endpoint (`POST /api/v1/train`) so the
  system goes dataset → served model without a CLI step; central fit is
  the default serving path, federated FedAvg available per request; n8n
  stays orchestration-only and drives the lifecycle in one workflow
  (`n8n/healthcare-endtoend.json`).
- ADR-012: image analysis reuses the same service boundary as tabular —
  `preprocessing.image` → `models.image` → `ClinicalCrew` image branch →
  same `ClinicalReport`; `ImageClassifier` maps string labels by
  `np.unique` order; `POST /api/v1/analyze/image` takes base64 JSON
  (decoded by a Pydantic validator) rather than multipart; the CNN is
  trained offline (`scripts/train_image_model.py`) since CNN training is
  too slow for an HTTP request; dashboard forms are driven by
  `GET /api/v1/model` so they adapt to the served model (raw JSON
  fallback when no model configured).
- ADR-013: privacy is opt-in per federated train request. Opacus DP-SGD
  needs a `torch.nn.Module`, so the DP path uses `TorchMLPClassifier`
  (same `get_parameters`/`set_parameters` contract → client/server stay
  model-agnostic); default non-DP federated path keeps sklearn
  `TabularClassifier`. `SecureAggregator` requires equal weights (masks
  only cancel under a uniform coefficient — matches `FedAvgServer`
  weighting each client once). Opacus runs `secure_mode=False`
  (experimentation; production re-run needed). Anonymization /
  pseudonymization live in `federated/privacy.py` (they protect federated
  exchange payloads; raw frames never leave a client, so leakage is
  structurally zero).
- The old `CrewAI/app/*` demo was removed (superseded by
  `preprocessing/`, `models/`, `federated/`, `rag/`, and
  `CrewAI/orchestrator/`); the production CrewAI module is
  `CrewAI/orchestrator/`.

## Status

Milestone 9 (privacy-preserving federated learning, paper §8) complete:
the old demo's DP + SecAgg + anonymization + MIA/leakage metrics are
ported to `backend/federated/privacy.py`, wired into the federated
stack and the training API, tested (289 backend + 13 frontend), lint
clean, and live-verified. Milestone 8.1 (image analysis + friendly
dashboard) remains complete and live-verified. All of this work is
uncommitted, awaiting user go-ahead to commit + push. Open question:
`gemini-3.7-flash` (default) returns transient 503s; user may prefer
switching the default to `gemini-3.6-flash`. Next milestones: real flwr
deployment path, then API hardening.