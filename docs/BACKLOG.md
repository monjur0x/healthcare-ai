# Milestone 1 — Preprocessing (complete)

## Preprocessing

### Scaffolding

- [x] Package structure

- [x] Config

- [x] Exceptions

- [x] Logger

### CSV

- [x] CSV Validator

- [x] CSV Cleaner

- [x] Missing Value Imputer

- [x] Encoder

- [x] Feature Engineering

- [x] Scaler

- [x] Transformer

- [x] Pipeline

- [x] Unit tests

### Image

- [x] Validator

- [x] Loader

- [x] Augmentation

- [x] Normalization

- [x] Pipeline

### Multimodal

- [x] Fusion

- [x] Metadata

---

# Milestone 2 — Models

Scope derived from `ai-automation-research.md` (§11 Proposed Methodology)
and `workflow.txt` (Phase 3): models consume preprocessing outputs, are
trained locally per hospital, and are aggregated by Flower (FedAvg).

## Shared

- [x] Model interface (`models/base.py`) — fit / predict / predict_proba / save / load
- [x] Model exceptions (`models/exceptions.py`)
- [x] Fixed seeds for reproducibility
- [x] Unit tests

## CSV / tabular

- [x] `TabularClassifier` (sklearn: gradient boosting / logistic / MLP)
- [x] Consume `CSVPipeline` output directly (accepts preprocessed DataFrame)
- [x] Persistence to `artifacts/` via joblib
- [x] Unit tests
- [x] End-to-end CSV → FedAvg demo (`examples/fedavg_demo.py`, presets
      for diabetes / heart / kidney / sepsis datasets)

## Image

- [x] `ImageClassifier` (torch CNN: conv → batch-norm → pool → MLP head)
- [x] Consumes channels-last `(N, H, W, C)` `ImageResult`-style batches
- [x] Fixed seed + deterministic dataloader for reproducibility
- [x] Persistence via `torch.save` / `load`
- [x] Unit tests
- [ ] Pretrained backbones (EfficientNetV2 / DenseNet / Swin-T) —
      deferred; current CNN is dependency-light and offline-friendly

## Multimodal

- [x] `FusionClassifier` consuming `FusionResult` from preprocessing
- [x] MLP (sklearn) over the fused feature matrix by default
- [x] Unit tests

## Evaluation hooks

- [x] `evaluation/metrics.py` — `ClassificationMetrics` + `classification_metrics`
- [x] `evaluate_classifier(model, X, y_true)` scores any fitted `BaseModel`
- [x] Accuracy, precision/recall/F1 (macro), MCC, ROC-AUC, PR-AUC, log loss
- [x] Unit tests
- [x] Federated metrics — communication cost (`parameter_set_bytes`,
      per-round + total bytes), convergence (`round_accuracy_deltas`,
      `convergence_round`), training time (per-round + total) via
      `federated/metrics.py` and the `server.metrics` property
- [ ] Privacy budget metrics — deferred (no privacy mechanism yet)
- [ ] Differential privacy — the removed old demo
      (`backend/CrewAI/app/federated/privacy.py`) contained a DP module
      (noise-multiplier approach, epsilon/delta targets) that was never
      ported to the new `federated/` module; port it when a privacy
      mechanism is scoped

## Federated

- [x] `federated/parameters.py` — `average_weights` (FedAvg), NumPy-native
- [x] `federated/client.py` — `FederatedClient` (flwr `NumPyClient`,
      flwr 1.33.0) with warm start, one local partial-fit per round,
      log-loss + accuracy evaluation
- [x] Weight exchange on models: `get_parameters` / `set_parameters`
      (tabular logistic/MLP, fusion, CNN via state dict);
      `partial_fit` (MLP)
- [x] `federated/server.py` — synchronous `FedAvgServer` driver
      (init global weights → per-round client fit → aggregate →
      evaluate) + `make_global_evaluator`; mirrors flwr `FedAvg`
      without the Ray process spawn (hermetic)
- [x] Unit tests (roundtrip, FedAvg round, client + server evaluate)
- [x] Federate the CNN end-to-end via `ImageClassifier.partial_fit`
      (one-epoch incremental training, ADR-006)
- [ ] Run the driver against real flwr `run_simulation` / a networked
      `ServerApp` for deployment (blocked: `ray` not installed; flwr
      simulation uses a Ray backend)

---

# Milestone 3 — RAG (complete)

Scope derived from `docs/SOFTWARE_ARCHITECTURE.md` §rag/: document
ingestion, embedding generation, vector search, and context retrieval
from PubMed / WHO / CDC / NICE / hospital-protocol knowledge sources.

## Scaffolding

- [x] Exceptions (`rag/exceptions.py`)
- [x] Configuration (`rag/config.py`) — `RAGSettings`, env prefix `RAG_`
- [x] Data structures (`rag/documents.py`) — `Document` / `Chunk` /
      `RetrievalResult`
- [x] Unit tests (`rag/tests/test_chunker.py`, `test_embedder.py`,
      `test_vector_store.py`, `test_retriever.py`, `test_rag_metrics.py`,
      `test_rag_pipeline.py`) — 38 passing

## Retrieval

- [x] `TextChunker` — deterministic word-based sliding window with
      configurable overlap
- [x] `Embedder` ABC + `TfidfEmbedder` (corpus-fitted, default) +
      `HashingEmbedder` (fit-free fixed-dim) + `build_embedder`
- [x] `VectorStore` — in-memory NumPy nearest-neighbour (cosine / dot)
- [x] `Retriever` — incremental ingest, query → top-k chunks,
      `build_context` (source-labelled prompt block)
- [x] `RetrievalMetrics` — `precision_at_k`, `recall_at_k`, MRR
- [x] `RAGPipeline` — chunker → embedder → store → retriever
- [x] Demo (`examples/rag_demo.py`) — corpus → ingest → queries →
      context + quality metrics; smoke tests (2 passing)

## Deferred

- [ ] Dense / transformer embeddings (sentence-transformers, Qdrant,
      GPU) — deferred; TF-IDF + in-memory store keep the module
      dependency-light and offline-friendly (ADR-007)
- [ ] Streaming ingestion from PubMed / WHO / CDC / NICE APIs
- [ ] Hybrid retrieval (BM25 + dense) and re-ranking

---

# Milestone 4 — CrewAI Orchestration (complete)

Scope derived from `docs/SOFTWARE_ARCHITECTURE.md` §CrewAI/: agents
orchestrate reasoning over the outputs of preprocessing, prediction,
and retrieval modules; they never implement ML.

## Scaffolding

- [x] `orchestrator/config.py` — `CrewSettings` (env prefix `CREW_`)
- [x] `orchestrator/exceptions.py` — `CrewError` + subclasses
- [x] `orchestrator/schemas.py` — `PatientInfo`, `PredictionResult`,
      `RiskResult`, `EvidenceItem`, `ClinicalReport`

## Deterministic services

- [x] `run_prediction` — single-row prediction from a fitted model
- [x] `assess_risk` — risk score/level + monitoring schedule
- [x] `retrieve_evidence` — RAG evidence wrapper
- [x] `assemble_clinical_report` — final structured report

## CrewAI layer

- [x] `prompts.py` — seven agent profiles + task descriptions + report schema
- [x] `tools.py` — `PredictionTool`, `RiskAssessmentTool`,
      `RAGRetrievalTool`, `ClinicalReportTool`
- [x] `agents.py` / `tasks.py` — seven agents and chained tasks
- [x] `crew.py` — `ClinicalCrew` (deterministic + optional LLM path,
      ADR-008)
- [x] Unit tests — 25 passing (hermetic, no LLM keys)
- [x] Demo (`examples/clinical_crew_demo.py`) + smoke tests (2 passing)

## Deferred

- [ ] Wire the LLM path to an installed provider (crewai google/OpenAI
      extra); requires a provider API key (never commit secrets)
- [ ] Image-path analysis through the crew (`input_type="image"` with
      `ImageClassifier` outputs)
- [ ] Crew memory / long-term storage of past reports

---

## Backlog

### Examples

- [x] `fedavg_demo.py` — CSV → `CSVPipeline` → MLP → FedAvg + report
- [x] `image_fedavg_demo.py` — image folders → `ImagePipeline` → CNN →
      FedAvg + report (smoke-tested on synthetic trees)
- [x] `rag_demo.py` — corpus directory → `RAGPipeline` → queries →
      context + quality metrics (smoke-tested on synthetic corpus)

### Preprocessing enhancements

- [ ] Add datetime parsing to feature engineering.
- [ ] Add robust error-reporting structure to `CSVPipeline` (~ `valid_frame` etc.).
- [ ] Expose a CLI or fit/persist for scaler/encoder parameters (reproducibility).
- [ ] Persist image normalization statistics (mean/std) for inference-time
      consistency (currently stateless fallback for `standard` mode).
- [ ] Add DICOM unit tests; requires adding `pydicom` to dependencies.
- [ ] Consider aspect-ratio-preserving (letterbox) resize option in
      `ImageLoader` (currently exact square resize).

### Milestone 5 — FastAPI API (complete)

Scope derived from `docs/SOFTWARE_ARCHITECTURE.md` §api/: request
validation, authentication, response serialization; business logic stays
in services (never in routes).

- [x] `api/config.py` — `APISettings` (env prefix `API_`)
- [x] `api/exceptions.py` — `APIError` hierarchy (status/code)
- [x] `api/schemas.py` — request models + `HealthResponse`; responses
      reuse orchestrator schemas
- [x] `api/services.py` — `AnalysisService` facade + load/build helpers
- [x] `api/routes.py` — `/api/v1/predict`, `/api/v1/retrieve`,
      `/api/v1/analyze` (validation + delegation only; optional bearer
      token auth)
- [x] `api/main.py` — `create_app()` factory + uvicorn entry point
- [x] Unit tests (`api/tests/`) — 19 passing
- [x] `fastapi` / `uvicorn[standard]` added to `backend/requirements.txt`

### Milestone 6 — n8n orchestration (complete)

Scope derived from `docs/SOFTWARE_ARCHITECTURE.md` §n8n/: orchestration
only — n8n triggers workflows and calls the FastAPI backend; AI
reasoning stays in the CrewAI crew (see `AGENTS.md`).

- [x] `n8n/clinical-analysis.json` — webhook (`/webhook/healthcare-analyze`)
      → `POST /api/v1/analyze` → validate + summarize → structured
      `status: success|error` response
- [x] `n8n/clinical-pipeline-modality.json` — webhook
      (`/webhook/healthcare-pipeline`) → normalize → switch on
      `modality` (image / csv) → `/api/v1/analyze` with matching
      `input_type` → merged summary or merged errors
- [x] Optional bearer-token support via an `httpHeaderAuth` credential
      (placeholder reference; backend `API_TOKEN` off by default)
- [x] `n8n/README.md` — import, configuration, example payloads, local
      smoke test
- [x] Removed the stale `n8n/workflow.json` / `workflow2.json` (they
      targeted the removed old demo endpoints `/predict/federated`,
      `/predict/image`, `/agents/run-crew`)

### Milestone 7+ (not yet scoped)

- [ ] Full OAuth / per-user authentication (currently an optional static
      bearer token via `API_TOKEN`)
- [ ] File upload endpoint for CSV / image inference
- [ ] Deployment container for the API (the old app's Dockerfile was
      removed with the superseded demo)
- [ ] Downstream n8n storage/notification branches (e.g. append report
      to a local file, Slack/Discord notify) using real credentials
- [ ] Docker Compose profile that runs n8n + FastAPI + Qdrant together