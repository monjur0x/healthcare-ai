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
- [x] Privacy budget metrics — `FederatedMetrics` now carries
      `secure_aggregation`, `differential_privacy`, and `epsilon`
      (worst-case per-client epsilon) alongside the cost/convergence
      fields
- [x] Differential privacy + secure aggregation + anonymization —
      ported the removed old demo (`backend/CrewAI/app/federated/
      privacy.py`) into `backend/federated/privacy.py` (paper Section 8):
      `anonymize_frame`, `pseudonymize`, `train_with_differential_privacy`
      (Opacus DP-SGD + epsilon audit), `SecureAggregator` (pairwise
      one-time-pad, adapted to the repo's `list[np.ndarray]` param
      format), `membership_inference_auroc`, `data_leakage_rate`, and
      `privacy_metrics_summary` (epsilon, budget-used %, MIA-AUROC,
      attack-resistance score, leakage rate, mechanisms). Wired into
      `FederatedClient` (local DP-SGD via `TorchMLPClassifier`),
      `FedAvgServer` (optional secure aggregation), and the API
      (`POST /api/v1/train` → `federated_metrics.privacy`)

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

- [x] Dense / transformer embeddings (sentence-transformers) — done in
      Milestone 10 via `SentenceTransformerEmbedder` (opt-in
      `RAG_EMBEDDING_MODEL=sentence-transformer`); TF-IDF stays the
      default. Persistent backend done via ChromaDB (`RAG_VECTOR_STORE`)
- [ ] Hybrid retrieval (BM25 + dense) and re-ranking
- [ ] Vector-store backends beyond ChromaDB (e.g. Qdrant) behind
      `build_vector_store()`
- [ ] Streaming ingestion from PubMed / WHO / CDC / NICE APIs

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

### Milestone 7 — Streamlit dashboard (reworked in Milestone 11)

Scope: a research-facing frontend for the FastAPI backend. The dashboard
is a thin view layer; all reasoning stays server-side (ADR-010).

- [x] `frontend/dashboard/client.py` — `HealthcareAPIClient` (httpx):
      `health` / `predict` / `retrieve` / `analyze`, optional bearer
      token, typed `HealthcareAPIError`
- [x] `frontend/dashboard/tests/test_client.py` — passing
      (`httpx.MockTransport`, hermetic)
- [x] `frontend/streamlit_app.py` — sidebar config + live health
      indicator; tabs for Clinical Analysis, Prediction, Evidence
      Retrieval, and Info (**replaced by the Milestone 11 CDS layout**)
- [x] `frontend/dashboard/tests/test_app_smoke.py` — passing
      (Streamlit AppTest: boots clean; mocked analyze submission renders
      the report)
- [x] `frontend/requirements.txt` — `streamlit`, `httpx`, `pytest`
- [x] Live end-to-end verified against a running backend

### Milestone 8 — Functional end-to-end system (complete)

Scope: connect the trained models, API, n8n, and dashboard into one
automated system that runs on CPU-only hardware.

- [x] `POST /api/v1/train` — train a model through the API (preset or
      dataset+target; central or federated FedAvg path) and serve it
      immediately; returns artifact path + hold-out metrics
- [x] `n8n/healthcare-endtoend.json` — single workflow automating the
      full lifecycle: webhook → train → analyze → write report to disk →
      structured response (replaces `clinical-pipeline-modality.json`)
- [x] `scripts/run_system.sh` — one-command start/status/stop (backend +
      dashboard + n8n Docker)
- [x] Root `README.md` — step-by-step run guide
- [x] Verified live on diabetes: central + federated train, predict,
      retrieve, analyze (report + risk + evidence)

### Milestone 8.1 — Image analysis + friendly dashboard input (complete)

- [x] `POST /api/v1/analyze/image` (base64 image → brain-tumor CNN →
      report) and `GET /api/v1/model` (feature columns for forms)
- [x] `scripts/train_image_model.py` + brain-tumor artifact trained
      (4 classes, ~0.71 accuracy / ~0.92 ROC-AUC)
- [x] Friendly per-feature inputs + MRI upload mode in the dashboard
- [x] Verified live on a real glioma scan

### Milestone 11 — Doctor-facing CDS dashboard (complete)

- [x] `frontend/dashboard/clinical.py` — grouped inputs, display labels,
      feature bounds, flag detection, payload builder, post-hoc pipeline
      stages, explainable decision sections (+14 tests)
- [x] Five-tab CDS layout (Overview / Clinical Assessment / Imaging /
      Results / System Status) with a model-driven assessment form and
      one **Analyze Patient** action
- [x] n8n routing: `analyze_via_n8n()` reads the full report from the
      end-to-end webhook (Code node returns `report`); Automatic /
      Via-n8n / Direct route selection; `n8n_health()` probe (+6 tests)
- [x] Honest rendering of unsupported outputs (mortality / readmission
      risk = "not estimated"; no fabricated treatment recommendations)
- [x] `frontend/pyproject.toml` mirrors backend tooling config
- [x] Tests: frontend 35 (+22), backend 326 (unchanged); lint clean;
      live-verified analyze + analyze_image against a running backend

New deferred items from Milestone 11:

- [ ] Persistent patient records + history in the dashboard (each
      assessment is currently entered fresh)
- [ ] Backend mortality-risk and readmission-risk models so the Results
      page can report them instead of "not estimated" (currently the
      `ClinicalReport` has no such fields)
- [ ] Backend feature-importance / SHAP-style explainability so the
      "Explainable Decision Report" can be model-derived rather than
      derived from prediction / risk outputs
- [ ] Live n8n end-to-end verification from the dashboard (no local n8n
      instance during the Milestone 11 session; covered by hermetic
      tests + workflow JSON validation)

### Milestone 9+ (not yet scoped)

- [ ] Full OAuth / per-user authentication (currently an optional static
      bearer token via `API_TOKEN`)
- [ ] File upload endpoint for CSV / image inference (backend) + upload
      widget in the dashboard
- [ ] Deployment container for the API (the old app's Dockerfile was
      removed with the superseded demo)
- [ ] Downstream n8n storage/notification branches (e.g. append report
      to a local file, Slack/Discord notify) using real credentials
- [ ] Docker Compose profile that runs n8n + FastAPI + Qdrant together
- [ ] Next.js dashboard (architecture doc names it; Streamlit currently
      fills the frontend role)
- [ ] Real flwr `run_simulation` / networked `ServerApp` (blocked:
      `ray` not installed)
- [ ] Opacus `secure_mode=True` for the production DP re-training pass
      (currently `secure_mode=False` for experimentation speed, with a
      UserWarning asking for one final secure retrain before release)

### Milestone 10 — Evaluation-gap closure (complete)

- [x] `opacus>=1.5.0` declared in `backend/requirements.txt`
- [x] ChromaDB persistent vector store (`RAG_VECTOR_STORE=chroma`) via
      `ChromaVectorStore` + `build_vector_store()` factory
- [x] Dense embeddings (`RAG_EMBEDDING_MODEL=sentence-transformer`,
      `SentenceTransformerEmbedder`, default `BAAI/bge-small-en-v1.5`)
- [x] RAGAS-style metrics: `context_precision`, `context_recall`,
      `faithfulness`, `answer_relevancy` + `RAGQualityMetrics`
- [x] Agent metrics: `task_completion_rate`, `decision_consistency`,
      `agent_collaboration_score` + `AgentMetrics` wired into
      `ClinicalReport.agent_metrics`
- [x] ADR-014 (transport-layer TLS/mTLS) + README "Privacy & Security"
- [x] Tests: backend 326 (+37), frontend 13; lint clean

New deferred items from Milestone 10:

- [ ] Wire RAGAS / agent metrics into the API response or a standalone
      evaluation endpoint (currently library functions + tests only)
- [ ] RAGAS-vs-heuristic calibration: compare the LLM-free faithful/\
      relevancy proxies against a judge-LLM baseline on a labeled set
- [ ] Baseline comparison study (paper §13): centralized vs federated vs
      federated+RAG vs federated+multi-agent on the shipped datasets