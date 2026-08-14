# Development Status

## Legend

- [x] Implemented and tested.
- [ ] Not started.

---

## Milestone 1 — Preprocessing

### Package scaffolding

- [x] Package structure (`backend/preprocessing/__init__.py`)
- [x] Global configuration (`config.py`)
- [x] Centralized logging (`logger.py`)
- [x] Custom exceptions (`exceptions.py`)

### CSV preprocessing (`backend/preprocessing/csv`)

- [x] Validator (`validator.py`)
- [x] Cleaner / column normalization (`cleaner.py`)
- [x] Missing value imputation (`imputer.py`)
- [x] Categorical encoding (`encoder.py`)
- [x] Feature engineering (`feature_engineering.py`)
- [x] Scaling (`scaler.py`)
- [x] Transformer / pipeline orchestration (`transformer.py`)
- [x] High-level entry point (`pipeline.py`)
- [x] Unit tests (`tests/test_csv.py`) — 27 passing

### Image preprocessing (`backend/preprocessing/image`)

- [x] Validator (`validator.py`)
- [x] Loader (`loader.py`) — PNG/JPG via Pillow, DICOM via optional `pydicom`
- [x] Augmentation (`augmentation.py`) — deterministic, seeded
- [x] Normalization (`normalization.py`) — minmax / zero_mean / standard
- [x] Pipeline (`pipeline.py`) — load → validate → resize → augment → normalize
- [x] Convenience API (`preprocessing.py`) — single image, batch, directory
- [x] Unit tests (`tests/test_image.py`) — 31 passing

### Multimodal preprocessing (`backend/preprocessing/multimodal`)

- [x] Fusion (`fusion.py`) — concatenate + summary/flatten image reduction
- [x] Metadata (`metadata.py`) — `SampleMetadata` / `ImageInfo` schemas
- [x] Unit tests (`tests/test_multimodal.py`) — 12 passing

---

## Tooling

- [x] `backend/pyproject.toml` with shared Black / Ruff / isort settings
- [ ] Lint/format/test commands documented before every session (see `AGENTS.md`)

---

## Not yet planned

Milestones for `federated/`, `rag/`, `evaluation/`, CrewAI orchestration,
and the FastAPI `api/` are complete; `n8n/` is not yet scoped in the
backlog.

---

## Milestone 5 — FastAPI API (`backend/api`)

### Scaffolding

- [x] `config.py` — `APISettings` (env prefix `API_`): server metadata,
      `MODEL_PATH`, `CORPUS_DIR`, optional `API_TOKEN`, CORS origins
- [x] `exceptions.py` — `APIError` + `ServiceUnavailableError` /
      `InvalidInputError` / `AuthenticationError` / `NotFoundError`
- [x] `schemas.py` — request models (`PredictRequest`, `RetrieveRequest`,
      `AnalyzeRequest`, `HealthResponse`) reusing orchestrator
      `PredictionResult` / `EvidenceItem` / `ClinicalReport` responses

### Service layer

- [x] `services.py` — `AnalysisService` facade: lazy model load,
      RAG corpus ingest (directory or built-in corpus), deterministic
      crew analysis; domain exceptions translated to typed `APIError`s
- [x] `load_predictive_model` / `build_rag_pipeline` helpers

### Routes (validation + delegation only)

- [x] `routes.py` — `/api/v1/predict`, `/api/v1/retrieve`,
      `/api/v1/analyze`; optional bearer-token auth (router dependency)
- [x] `main.py` — `create_app()` factory (DI via app state, CORS,
      `APIError` → JSON handler); module-level `app` for uvicorn
- [x] Unit tests (`tests/test_api.py`, `tests/test_services.py`) —
      19 passing
- [ ] OAuth / full user authentication (currently an optional static
      bearer token; see backlog)

---

## Milestone 2 — Models

### Shared (`backend/models`)

- [x] Model interface (`base.py`) — fit / predict / predict_proba / save / load
- [x] Model exceptions (`exceptions.py`)
- [x] Unit tests (`models/tests/test_tabular.py`)

### CSV / tabular (`backend/models/csv`)

- [x] `TabularClassifier` (`tabular.py`) — gradient boosting / logistic / MLP
- [x] Persistence via joblib
- [x] Unit tests — 10 passing

### Image (`backend/models/image`)

- [x] `ImageClassifier` (`cnn.py`) — torch CNN, trains/infers on
      channels-last `(N, H, W, C)` batches
- [x] Adaptive pooling CNN: conv → batch-norm → pool → MLP head
- [x] `partial_fit` — one-epoch incremental training from current
      weights; the image path joins federated rounds (ADR-006)
- [x] Deterministic training (seeded RNG + seeded dataloader shuffle)
- [x] Persistence via `torch.save` / `ImageClassifier.load`
- [x] Unit tests (`models/tests/test_cnn.py`) — 16 passing

### Multimodal (`backend/models/multimodal`)

- [x] `FusionClassifier` (`fusion_model.py`) consuming `FusionResult`
      directly (or raw fused matrix); MLP over fused features
- [x] Composes `TabularClassifier` (DRY), joblib persistence
- [x] Unit tests (`models/tests/test_fusion_model.py`) — 10 passing

### Model configuration (`backend/models/config.py`)

- [x] `ModelSettings` — seed, image epochs / batch size / learning rate /
      device; env prefix `MODEL_`

### Evaluation (`backend/evaluation`)

- [x] `metrics.py` — `ClassificationMetrics` dataclass (accuracy,
      precision/recall/F1 macro, MCC, ROC-AUC, PR-AUC, log loss)
- [x] `classification_metrics(y_true, y_pred, y_score, labels)` — pure
      function, binary + multiclass, graceful None for undefined metrics
- [x] `evaluate_classifier(model, X, y_true)` — uniform scoring of any
      fitted `BaseModel` (tabular / image / fusion)
- [x] Unit tests (`tests/test_metrics.py`) — 11 passing

### Federated (`backend/federated`)

- [x] `parameters.py` — `average_weights` (element-wise FedAvg)
- [x] `client.py` — `FederatedClient` (flwr 1.33 `NumPyClient`): warm
      start, one local `partial_fit` per round, log-loss + accuracy eval
- [x] Weight exchange on models (`get_parameters` / `set_parameters`)
      for tabular logistic/MLP, fusion, and CNN; `partial_fit` for MLP;
      `set_parameters` materializes unfitted estimators via dummy fit
- [x] `server.py` — synchronous `FedAvgServer` (init weights, per-round
      client fit, aggregate, evaluate) + `make_global_evaluator`;
      mirrors flwr `FedAvg` without the Ray process spawn
- [x] Unit tests (`tests/test_parameters.py`, `test_client.py`,
      `test_server.py`, `test_cnn_federation.py`) — 25 passing
- [x] CNN federates end-to-end via `ImageClassifier.partial_fit`
      (ADR-006); end-to-end tests in `federated/tests/`

### Federated metrics (`federated/metrics.py`)

- [x] `parameter_set_bytes` — bytes for a full weight exchange
- [x] `round_accuracy_deltas` + `convergence_round` — round-to-round
      accuracy change and first converged round (threshold-tunable)
- [x] `FederatedMetrics` dataclass (rounds/clients, bytes exchanged,
      per-round + total time, accuracy deltas, convergence round) with
      `to_dict()` for JSON reports
- [x] `FedAvgServer.run()` records per-round wall-clock duration and
      estimated communication bytes (client upload + broadcast);
      exposed via `RoundResult` fields and the `server.metrics` property
- [x] Demo report (`fedavg_demo.py`) now includes `federated_metrics`
- [x] Unit tests (`tests/test_federated_metrics.py`) + server tests —
      10 passing

### End-to-end demo (`backend/examples`)

- [x] `fedavg_demo.py` — CSV → `CSVPipeline` → `TabularClassifier`
      (MLP) → FedAvg rounds → evaluation report. Presets for the local
      datasets: diabetes / heart / kidney / sepsis. Partitions train
      rows into class-balanced client shards (StratifiedKFold), trains
      the synchronous `FedAvgServer`, compares against a central
      baseline, writes `global_model.joblib` + `report.json`.
- [x] `image_fedavg_demo.py` — image-path FedAvg demo. Discovers
      class-labelled image folders (e.g. the brain-tumor MRI dataset),
      preprocesses with `ImagePipeline`, federates the CNN via
      `ImageClassifier.partial_fit`, reports baseline + federated
      metrics, writes `global_model.pt` + `report.json`.
- [x] Smoke tests (`examples/tests/test_image_fedavg_demo.py`) — 3
      passing (run on synthetic image trees, no external data)

---

## Milestone 3 — RAG (Retrieval-Augmented Generation)

Scope: document ingestion, embedding generation, vector search, and
context retrieval, per `docs/SOFTWARE_ARCHITECTURE.md` §rag/.

### Package scaffolding (`backend/rag`)

- [x] Exceptions (`exceptions.py`) — `RAGError` + `EmptyCorpusError`,
      `EmptyQueryError`, `InvalidDocumentError`, `EmbeddingError`,
      `RetrievalError`
- [x] Configuration (`config.py`) — `RAGSettings` (env prefix `RAG_`):
      chunk size/overlap, embedding model, max features, top-k, metric
- [x] Data structures (`documents.py`) — `Document` / `Chunk` /
      `RetrievalResult`, all frozen dataclasses with `to_dict()`

### Retrieval components

- [x] `chunker.py` — `TextChunker`: deterministic word-based sliding
      window with configurable overlap
- [x] `embedder.py` — `Embedder` ABC + `TfidfEmbedder` (corpus-fitted)
      + `HashingEmbedder` (fit-free fixed-dim) + `build_embedder`;
      transformer embedders swappable behind the interface
- [x] `store.py` — `VectorStore`: in-memory NumPy nearest-neighbour
      search over cosine / dot
- [x] `retriever.py` — `Retriever`: incremental ingest, query → top-k
      chunks, `build_context` (source-labelled prompt block)
- [x] `metrics.py` — `precision_at_k`, `recall_at_k`,
      `mean_reciprocal_rank`, `RetrievalMetrics`
- [x] `pipeline.py` — `RAGPipeline` composing chunker → embedder →
      store → retriever (`ingest_documents`, `ingest_texts`,
      `retrieve`, `build_context`)
- [x] No new dependencies (reuses scikit-learn)
- [x] Unit tests (`rag/tests/`) — 38 passing

### Retrieval demo (`backend/examples`)

- [x] `rag_demo.py` — corpus directory → `RAGPipeline` → queries →
      top-k chunks + context + (optional) quality metrics
- [x] Smoke tests (`examples/tests/test_rag_demo.py`) — 2 passing

---

## Milestone 4 — CrewAI Orchestration

Scope: multi-agent reasoning over the outputs of the preprocessing,
prediction, and retrieval modules, per `docs/SOFTWARE_ARCHITECTURE.md`
§CrewAI/.

### Package scaffolding (`backend/CrewAI/orchestrator`)

- [x] Configuration (`config.py`) — `CrewSettings` (env prefix `CREW_`):
      optional LLM provider/model/key, crew verbosity/memory, RAG top-k,
      risk + marker thresholds
- [x] Exceptions (`exceptions.py`) — `CrewError` + tool / report /
      `LLMNotConfiguredError` subclasses
- [x] Schemas (`schemas.py`) — `PatientInfo`, `PredictionResult`,
      `RiskResult`, `EvidenceItem`, `ClinicalReport` (pydantic)

### Deterministic services (`services.py`)

- [x] `run_prediction` — single-row prediction from a fitted
      `TabularClassifier` (feature-aligned, error-guarded)
- [x] `assess_risk` — risk score/level from confidence + clinical
      marker thresholds, with a deterministic monitoring schedule
- [x] `retrieve_evidence` — wraps `RAGPipeline` into `EvidenceItem`s
- [x] `assemble_clinical_report` — final structured report, consistent
      prediction+risk pairing enforced

### CrewAI layer

- [x] `prompts.py` — role/goal/backstory for the seven agents, task
      descriptions, report schema instructions
- [x] `tools.py` — `PredictionTool`, `RiskAssessmentTool`,
      `RAGRetrievalTool`, `ClinicalReportTool` (crewai `BaseTool`
      wrappers over the services)
- [x] `agents.py` — seven agents (Patient Analysis, Disease Prediction,
      Medical Research, Treatment Planning, Explainability, Risk
      Monitoring, Report Writing)
- [x] `tasks.py` — chained tasks (analysis → prediction → evidence →
      treatment → explanation → risk → report)
- [x] `crew.py` — `ClinicalCrew`: `run_analysis()` offline
      deterministic pipeline (ADR-008), `run_llm()` optional CrewAI
      kickoff with deterministic fallback, `run()` selects by config
- [x] Agents/tasks/crew construct without an LLM key (hermetic)
- [x] Unit tests (`CrewAI/orchestrator/tests/`) — 25 passing

### Orchestration demo (`backend/examples`)

- [x] `clinical_crew_demo.py` — CSV → `CSVPipeline` →
      `TabularClassifier` → `ClinicalCrew.run_analysis()` → clinical
      report (`report.json`); built-in or `--corpus-dir` knowledge base
- [x] Smoke tests (`examples/tests/test_clinical_crew_demo.py`) —
      2 passing

### Cleanup (`backend/CrewAI`)

- [x] Removed the superseded old demo (`app/`, `tests/test_healthcare.py`,
      `Dockerfile`, `docker-compose.yml`, `requirements.txt`,
      `.env.example`, `README.md`) — nothing referenced it and the new
      modules supersede it
- [x] Untracked generated artifacts (`artifacts/*.pt`, `artifacts/*.json`)
      and ignored `artifacts/` via `.gitignore`
- [ ] Differential privacy port from the old demo (recorded in backlog)

---

## Testing

- [x] Preprocessing: 70 tests passing
- [x] Models: 36 tests passing (tabular 10 / CNN 16 / fusion 10)
- [x] Evaluation: 11 tests passing
- [x] Federated: 35 tests passing
- [x] RAG: 38 tests passing
- [x] Orchestration (CrewAI): 25 tests passing
- [x] Examples: 7 tests passing
- [x] Full suite: 222 tests passing (`pytest preprocessing/tests models/tests evaluation/tests federated/tests rag/tests examples/tests CrewAI/orchestrator/tests`)
- [ ] Full test command documented in README/AGENTS (see `AGENTS.md` tooling note)