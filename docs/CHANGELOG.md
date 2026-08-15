# Changelog

## [Unreleased]

### Added

- Draft project guide in `README1.md` covering architecture, quick
  start, backend/dashboard/n8n startup, dataset assumptions, and
  end-to-end execution steps for the complete healthcare AI system.

- `backend/preprocessing/csv/` module: `validator.py`, `cleaner.py`,
  `imputer.py`, `encoder.py`, `feature_engineering.py`, `scaler.py`,
  `transformer.py`, `pipeline.py`, and package `__init__.py`.
- Unit tests for the CSV preprocessing module (`tests/test_csv.py`,
  27 tests passing).
- `backend/pyproject.toml` with shared Black / Ruff / isort settings.
- `backend/preprocessing/image/` module: `validator.py`, `loader.py`,
  `augmentation.py`, `normalization.py`, `pipeline.py`,
  `preprocessing.py`, and package `__init__.py`.
- Image settings in `config.py`: resize, normalization mode (minmax /
  zero_mean / standard), mean/std defaults, augmentation flags.
- `ImageNormalizationError` and `ImageAugmentationError` exceptions.
- Unit tests for the image preprocessing module (`tests/test_image.py`,
  31 tests passing).
- `backend/preprocessing/multimodal/` module: `metadata.py` (sample and
  image metadata schemas), `fusion.py` (concatenate fusion with summary
  / flatten image reduction), and package `__init__.py`.
- Multimodal settings (`FUSION_MODE`, `FUSION_IMAGE_REDUCTION`) in
  `config.py` and `FusionError` exception.
- Unit tests for the multimodal preprocessing module
  (`tests/test_multimodal.py`, 12 tests passing).
- `backend/models/` module: `base.py` (abstract model interface),
  `exceptions.py`, and `csv/tabular.py` (`TabularClassifier` wrapping
  sklearn gradient boosting / logistic / mlp with joblib persistence).
- `backend/requirements.txt` covering backend dependencies.
- Unit tests for the models module (`models/tests/test_tabular.py`,
  10 tests passing).
- `backend/models/config.py` with `ModelSettings` (seed, image training
  hyperparameters, device; env prefix `MODEL_`).
- `backend/models/image/` module: `cnn.py` (`ImageClassifier`, a torch
  CNN with adaptive pooling) and package `__init__.py`. Accepts
  channels-last `(N, H, W, C)` or channels-first batches, trains
  deterministically with a seeded dataloader, persists via
  `torch.save` / `load`.
- `backend/models/multimodal/` module: `fusion_model.py`
  (`FusionClassifier` = MLP over `FusionResult.fused`, composing
  `TabularClassifier`) and package `__init__.py`.
- Unit tests for the new model modules (`models/tests/test_cnn.py`,
  12 passing; `models/tests/test_fusion_model.py`, 10 passing).
- Added `torch` to `backend/requirements.txt`.
- `backend/evaluation/` module: `metrics.py`
  (`ClassificationMetrics` dataclass, `classification_metrics` for
  accuracy / macro precision-recall-F1 / MCC / ROC-AUC / PR-AUC / log
  loss, and `evaluate_classifier` for uniform scoring of any fitted
  `BaseModel`), package `__init__.py`, and unit tests
  (`tests/test_metrics.py`, 11 passing).
- Weight exchange on models: `get_parameters` / `set_parameters`
  (tabular logistic/MLP, fusion, and CNN via torch state dict) and
  `partial_fit` (incremental MLP training) for federated learning.
- `backend/federated/` module: `parameters.py` (`average_weights`
  FedAvg aggregation), `client.py` (`FederatedClient`, a flwr 1.33
  `NumPyClient` with warm start and per-round local training), package
  `__init__.py`, and unit tests (`tests/test_parameters.py`,
  `tests/test_client.py`, 18 passing).
- `backend/federated/server.py` — synchronous `FedAvgServer` driver
  (initial global aggregation, per-round client fit / aggregate /
  evaluate) and `make_global_evaluator`; mirrors flwr `FedAvg`
  semantics without the Ray-based `run_simulation` process spawn so
  experiments stay hermetic. Unit tests (`tests/test_server.py`,
  5 passing). `RoundResult` gained `to_dict()` for JSON reports.
- `backend/examples/fedavg_demo.py` — end-to-end CSV → preprocessing →
  FedAvg demo. Loads a hospital CSV (presets: diabetes / heart /
  kidney / sepsis), runs `CSVPipeline`, partitions train rows into
  class-balanced client shards, trains an MLP with the synchronous
  `FedAvgServer`, and reports global metrics against a central
  baseline. Writes `global_model.joblib` and `report.json` to `--out`.
- `ImageClassifier.partial_fit` — one-epoch incremental CNN training
  from the current weights (labels restricted to fit-time classes), so
  the image path joins federated rounds via the existing
  `FederatedClient`/`FedAvgServer`; `BaseModel` now documents a default
  `partial_fit` raising `NotImplementedError`.
- End-to-end CNN federation tests (`federated/tests/test_cnn_federation.py`)
  plus `partial_fit` unit tests in `models/tests/test_cnn.py`.
- `federated/metrics.py` — `FederatedMetrics` + helpers
  (`parameter_set_bytes`, `round_accuracy_deltas`, `convergence_round`)
  for communication cost, convergence, and training time.
  `FedAvgServer.run()` records per-round wall-clock duration and
  estimated bytes exchanged (client upload + broadcast); `RoundResult`
  and the `server.metrics` property surface them, and the FedAvg demo
  report now includes a `federated_metrics` section.
- `examples/image_fedavg_demo.py` — end-to-end image-path FedAvg demo.
  Discovers class-labelled image folders, preprocesses batches with
  `ImagePipeline`, partitions train rows into class-balanced client
  shards, trains the CNN via `ImageClassifier.partial_fit` on the
  synchronous `FedAvgServer`, and reports baseline + federated metrics
  with a saved `global_model.pt`. Smoke tests in
  `examples/tests/test_image_fedavg_demo.py`.
- Added `flwr` to `backend/requirements.txt`.
- `backend/rag/` module: `exceptions.py` (`RAGError` + empty corpus /
  query, invalid document, embedding, retrieval subclasses),
  `config.py` (`RAGSettings`, env prefix `RAG_`), `documents.py`
  (`Document` / `Chunk` / `RetrievalResult`), `chunker.py`
  (`TextChunker`, word-based sliding window with configurable overlap),
  `embedder.py` (`Embedder` ABC + `TfidfEmbedder` + `HashingEmbedder`
  + `build_embedder`), `store.py` (`VectorStore`, NumPy cosine / dot
  nearest-neighbour), `retriever.py` (`Retriever` with incremental
  ingest and `build_context`), `metrics.py` (`precision_at_k`,
  `recall_at_k`, `mean_reciprocal_rank`, `RetrievalMetrics`), and
  `pipeline.py` (`RAGPipeline` composing chunker → embedder → store →
  retriever). No new dependencies (reuses scikit-learn).
- Unit tests for the RAG module (`rag/tests/test_chunker.py`,
  `test_embedder.py`, `test_vector_store.py`, `test_retriever.py`,
  `test_rag_metrics.py`, `test_rag_pipeline.py`) — 38 passing.
- `examples/rag_demo.py` — end-to-end retrieval demo. Loads a corpus
  directory of `.txt`/`.md` files, ingests them through `RAGPipeline`,
  answers queries with top-k chunks + a prompt-ready context block, and
  reports retrieval quality metrics (precision@k, recall@k, MRR) when a
  ground-truth JSON map is supplied. Writes `report.json` to `--out`.
- Smoke tests for the RAG demo (`examples/tests/test_rag_demo.py`) —
  2 passing (synthetic corpus, no external data).
- `backend/CrewAI/orchestrator/` — CrewAI multi-agent orchestration
  consuming the preprocessing / models / rag modules:
  - `config.py` — `CrewSettings` (env prefix `CREW_`): optional LLM
    provider/model/key, crew verbosity/memory/max-iter, RAG top-k, risk
    thresholds, clinical marker thresholds
  - `exceptions.py` — `CrewError` + tool / report / LLM-configuration
    subclasses
  - `schemas.py` — `PatientInfo`, `PredictionResult`, `RiskResult`,
    `EvidenceItem`, `ClinicalReport` (pydantic, serializable)
  - `services.py` — deterministic, LLM-free core: `run_prediction`,
    `assess_risk`, `retrieve_evidence` (wraps `RAGPipeline`),
    `assemble_clinical_report`
  - `prompts.py` — role/goal/backstory profiles for the seven agents,
    task descriptions, and the report schema instructions
  - `tools.py` — crewai `BaseTool` wrappers: `PredictionTool`,
    `RiskAssessmentTool`, `RAGRetrievalTool`, `ClinicalReportTool`
  - `agents.py` / `tasks.py` — the seven agents (Patient Analysis,
    Disease Prediction, Medical Research, Treatment, Explainability,
    Risk Monitoring, Report Writing) and their chained tasks; LLM is
    bound only on the LLM path so construction stays hermetic
  - `crew.py` — `ClinicalCrew`: `run_analysis()` (offline deterministic
    pipeline, ADR-008), `run_llm()` (CrewAI kickoff when
    `CREW_LLM_API_KEY` is set, with fallback to the deterministic
    report), `run()` picks by configuration
- Unit tests for the orchestrator (`CrewAI/orchestrator/tests/`
  `test_prediction_service.py`, `test_risk_service.py`,
  `test_report_service.py`, `test_tools.py`, `test_crew.py`,
  `test_agents.py`) — 25 passing (hermetic, no LLM keys).
- `examples/clinical_crew_demo.py` — end-to-end offline demo: CSV →
  `CSVPipeline` → `TabularClassifier` → `ClinicalCrew.run_analysis()`
  (prediction → risk → RAG evidence → clinical report) with a built-in
  or `--corpus-dir` knowledge base; writes `report.json`.
- Smoke tests for the clinical crew demo
  (`examples/tests/test_clinical_crew_demo.py`) — 2 passing.
- `backend/api/` module (Milestone 5, FastAPI):
  - `config.py` — `APISettings` (env prefix `API_`): server metadata,
    `MODEL_PATH`, `CORPUS_DIR`, optional `API_TOKEN`, CORS origins
  - `exceptions.py` — `APIError` + `ServiceUnavailableError` /
    `InvalidInputError` / `AuthenticationError` / `NotFoundError`
  - `schemas.py` — `PredictRequest`, `RetrieveRequest`,
    `AnalyzeRequest`, `HealthResponse`; responses reuse the
    orchestrator `PredictionResult` / `EvidenceItem` / `ClinicalReport`
  - `services.py` — `AnalysisService` facade (lazy model load + RAG
    corpus ingest + deterministic crew analysis) translating domain
    exceptions into typed `APIError`s; `load_predictive_model`,
    `build_rag_pipeline` with a default built-in medical corpus
  - `routes.py` — `/api/v1/predict`, `/api/v1/retrieve`,
    `/api/v1/analyze`; routes only validate + delegate, optional
    bearer-token auth via router dependency
  - `main.py` — `create_app()` factory (DI service via app state, CORS,
    `APIError` → JSON handler) + module-level `app` for uvicorn
- Unit tests for the API module (`api/tests/test_services.py`,
  `test_api.py`) — 19 passing (hermetic TestClient with a fake service
  plus a real fitted-model service test).
- Added `fastapi` / `uvicorn[standard]` to `backend/requirements.txt`.
- `n8n/` orchestration workflows (Milestone 6):
  - `n8n/clinical-analysis.json` — webhook (`healthcare-analyze`) →
    `POST /api/v1/analyze` → validate & summarize → structured
    `status: success|error` webhook response (error branch merges HTTP
    and validation failures)
  - `n8n/clinical-pipeline-modality.json` — webhook
    (`healthcare-pipeline`) → normalize input → switch on `modality`
    (image vs csv) → `/api/v1/analyze` with matching `input_type` →
    merged success summary or merged error payload
  - Optional backend bearer-token auth wired through an
    `httpHeaderAuth` credential (placeholder reference; works without
    the credential while `API_TOKEN` is unset)
  - `n8n/README.md` — import steps, configuration, example payloads,
    security notes, local smoke test
- `frontend/` Streamlit dashboard (Milestone 7):
  - `frontend/dashboard/client.py` — `HealthcareAPIClient` (httpx) over
    the FastAPI backend: `health` / `predict` / `retrieve` / `analyze`,
    optional bearer token, typed `HealthcareAPIError` (status + code)
    parsed from the backend error detail
  - `frontend/streamlit_app.py` — thin view layer: sidebar backend URL /
    token + live health indicator; tabs for Clinical Analysis (full
    report with prediction, risk, evidence, recommendations, JSON
    download), Prediction (probability bar chart), Evidence Retrieval
    (score bars), and Info (health + endpoint reference)
  - `frontend/dashboard/tests/` — 7 client tests (`httpx.MockTransport`)
    + 2 Streamlit AppTest smoke tests, all hermetic
  - `frontend/requirements.txt` — `streamlit`, `httpx`, `pytest`
- Verified the dashboard client live against a running uvicorn backend
  (`/health`, `/api/v1/analyze`, `/api/v1/retrieve`, and the 503
  response when no model is configured).
- `backend/api` — `POST /api/v1/train` endpoint (Milestone 8):
  - `api/schemas.py` — `TrainRequest` (preset or dataset+target,
    model family, test size, seed, federated flag + clients/rounds) and
    `TrainResponse` (artifact path, hold-out metrics)
  - `api/services.py` — `prepare_tabular_data` (CSV → pipeline →
    features/labels), `TrainResult`, `AnalysisService.train` (central
    fit or federated FedAvg path; the fitted model replaces the service
    model so `predict` / `analyze` use it immediately); `PRESETS`
    registry; `APISettings.ARTIFACTS_DIR` / `DATASET_DIR`
  - `api/routes.py` — `POST /api/v1/train` (validation + delegation only)
  - Tests: +14 (API suite now 33 passing; full backend suite 255)
- `n8n/healthcare-endtoend.json` — single end-to-end workflow:
  webhook → optional `/api/v1/train` → `/api/v1/analyze` → write report
  JSON to disk (`/tmp/healthcare_reports/`, override with `output_dir`)
  → structured `status: success|error` response. Replaces
  `clinical-pipeline-modality.json`; `clinical-analysis.json` kept as a
  minimal reference. `n8n/README.md` updated.
- `scripts/run_system.sh` — one-command `start` / `status` / `stop`:
  trains a default model via the API, starts the backend and dashboard,
  and starts n8n in Docker (`N8N_ENABLED=0` skips n8n).
- Root `README.md` — CPU-only step-by-step run guide (manual + one-command).
- **Image analysis + friendly dashboard input (Milestone 8.1):**
  - `models/image/cnn.py` — `ImageClassifier` accepts string class
    labels (class order from `np.unique` instead of `int(label)`)
  - `CrewAI/orchestrator/services.py` — `run_image_prediction` for
    preprocessed `(H, W, C)` images; `crew.py` gains the
    `image_model` / `image` analysis path
  - `api/` — `APISettings.IMAGE_MODEL_PATH`; `GET /api/v1/model`
    (`ModelInfo`); `POST /api/v1/analyze/image` (`AnalyzeImageRequest`
    with base64 validator); `AnalysisService.image_model` /
    `model_info()` / `analyze_image()`
  - `scripts/train_image_model.py` — trains the brain-tumor CNN and
    writes `backend/artifacts/brain/global_model.pt`; `run_system.sh`
    passes `API_IMAGE_MODEL_PATH`
  - `frontend/` — friendly per-feature numeric inputs (driven by
    `/api/v1/model`), image upload mode with preview, `model_info()` +
    `analyze_image()` client methods, model metadata in Info tab
  - Tests: backend 271 (+16), frontend 13 (+4)
- Default CrewAI LLM model bumped to `gemini-3.7-flash` (latest
  workhorse release) and `backend/.env.example` added documenting all
  API_ / CREW_ / MODEL_ / RAG_ variables incl. `CREW_LLM_API_KEY`;
  README gains a "Enabling the CrewAI LLM agents (Gemini)" section.
- All five settings classes now use `extra="ignore"` so a shared
  `backend/.env` with mixed prefixes (`API_` / `CREW_` / `MODEL_` /
  `RAG_` / `PREPROCESS_`) no longer breaks other configs.
- `ClinicalCrew.run_llm` exports `CREW_LLM_API_KEY` as `GEMINI_API_KEY`
  before kickoff (crewai 1.15's Gemini provider reads that env var);
  `crewai[google-genai]` installed. `test_run_prefers_deterministic_when_no_llm`
  is now hermetic (patches `LLM_API_KEY` to "").
- Verified: 7 agents / 7 tasks / crew construct; Gemini `gemini-3.6-flash`
  accepts calls and the crew returns the fallback report when the LLM
  output doesn't match the strict schema; `gemini-3.7-flash` was
  returning transient 503 "high demand" (model launched 2026-08-13).
- **Privacy-preserving federated learning (Milestone 9, paper §8):**
  - `federated/privacy.py` — `PrivacyConfig`, `anonymize_frame`
    (`PII_PATTERNS`), `pseudonymize`, `train_with_differential_privacy`
    (Opacus DP-SGD + epsilon audit), `SecureAggregator` (pairwise
    one-time-pad, adapted to `list[np.ndarray]`), `membership_inference_auroc`,
    `data_leakage_rate`, `privacy_metrics_summary`
    (epsilon / budget-used % / MIA-AUROC / attack-resistance score /
    leakage / mechanism); ported from the removed old demo's
    `CrewAI/app/federated/privacy.py`
  - `models/csv/TorchMLPClassifier` — torch MLP on the same
    `get_parameters` / `set_parameters` contract as `TabularClassifier`
    (Opacus needs a `torch.nn.Module`); saved as joblib payload
    `kind="torch_mlp"`
  - `federated/client.py` — `FederatedClient` accepts a `PrivacyConfig`;
    DP-SGD local training reports per-round epsilon
  - `federated/server.py` — `FedAvgServer(secure_aggregation=True)` uses
    the `SecureAggregator`; `FederatedMetrics` gains `secure_aggregation`,
    `differential_privacy`, `epsilon`
  - `api/` — `TrainRequest` gains `differential_privacy`,
    `noise_multiplier`, `max_grad_norm`, `privacy_delta`,
    `secure_aggregation`; `AnalysisService._train_federated` builds torch
    MLPs for DP, collects epsilons, runs the MIA audit (members = client
    shards, non-members = hold-out), and returns the
    `federated_metrics.privacy` block
  - Dependency: `opacus` (1.6.0) added to the backend venv
  - Live: diabetes preset, federated DP + SecAgg → ε ≈ 2.14 (53.5% of
    budget), MIA-AUROC ≈ 0.50, attack-resistance ≈ 1.0, leakage 0.0
  - Tests: backend 289 (+18), frontend 13

### Removed

- Removed the stale `n8n/workflow.json` and `n8n/workflow2.json`: they
  targeted the removed old demo endpoints (`/predict/federated`,
  `/predict/image`, `/agents/run-crew`, Streamlit dashboard, Qdrant
  direct search) that no longer exist after the CrewAI demo cleanup.
- Removed the superseded old CrewAI demo under `backend/CrewAI/`:
  `app/` (4,200 lines of orphaned demo API / crew / federated / models /
  rag / utils code), its `tests/test_healthcare.py` (one test failing),
  `Dockerfile`, `docker-compose.yml`, `requirements.txt`,
  `.env.example`, and `README.md`. Nothing referenced the old app; all
  of it is superseded by `preprocessing/`, `models/`, `federated/`,
  `rag/`, and `CrewAI/orchestrator/`.
- Untracked generated artifacts under `backend/CrewAI/artifacts/`
  (`global_model.pt`, `federation_summary.json`, `metrics.json`) and
  added `artifacts/` to `backend/CrewAI/.gitignore`; `backend/artifacts/`
  was already ignored.

### Changed

- Replaced sklearn-based scaling with a dependency-light NumPy
  implementation in `scaler.py`.
- Standardized tooling on `backend/pyproject.toml`.
- Aligned `[tool.isort]` `lines_between_types = 1` with the Ruff isort
  rule so both tools agree on import formatting.
- `TabularClassifier` and the new image/fusion models now read the
  random seed from `models.config` (`MODEL_RANDOM_SEED` via
  `models/config.py`) instead of `preprocessing.config`.
- `TabularClassifier.get_parameters` now returns interleaved
  `coefs_`/`intercepts_` (alternating W/b) so round-tripping through
  `set_parameters` is self-consistent for MLP and logistic.
- `TabularClassifier.set_parameters` now materializes an unfitted
  estimator with a deterministic dummy fit (structure only), so global
  weights can be injected into fresh models (used by the global
  evaluator); fitted estimators are validated for feature/count
  alignment.

### Fixed

- Normalized column handling so all stages operate on lowercase
  snake_case column names.
- Graceful skip of unavailable feature-engineering features instead of
  hard failure (configurable via `strict=True`).
- Import sorting in `logger.py` to satisfy both Ruff and isort.