# Changelog

## [Unreleased]

### Added

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

### Removed

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