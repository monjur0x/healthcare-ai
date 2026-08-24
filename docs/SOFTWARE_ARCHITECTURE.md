# Software Architecture

## Overview

The Healthcare AI Framework is a modular, service-oriented system for privacy-preserving clinical decision support. The system processes structured healthcare data (CSV) from four specialty hospitals (Diabetes / Heart / CKD / Sepsis), performs AI inference using machine learning models trained federatively across hospitals, coordinates reasoning through CrewAI agents, retrieves medical evidence using RAG, monitors longitudinal patient risk with escalation alerts, and automates clinical workflows using n8n.

The architecture is designed around separation of concerns. Each module has a single responsibility and communicates through well-defined interfaces.

---

# High-Level Architecture

```
                    Clinician
                        │
                        ▼
             Streamlit Dashboard  ←──── n8n Orchestration
                        │                       ▲
                        ▼                       │
                   FastAPI Backend ─────────────┘
                        │
        ┌───────────────┼────────────────┐
        ▼               ▼                ▼
  CSV Preprocessing  Image Preprocessing   Risk Monitor
        │               │                (trend + alerts)
        └───────┬───────┘                      ▲
                ▼                              │
        Prediction Models ─────────────────────┘
                │
                ▼
      CrewAI Multi-Agent System (7 agents)
                │
                ▼
           RAG Retrieval
                │
                ▼
       Clinical Report Generator
                │
                ▼
         FastAPI Response
                │
                ▼
    n8n Workflow → Storage • Doctor Notify • Dashboard
```

---

# Backend Structure

```
backend/

preprocessing/     # CSV / image pipelines, canonical loaders (no ML)
models/            # ML/DL models only (tabular MLP, torch MLP, image CNN)
federated/         # Flower FL: server, clients, hospitals, canonical schema
rag/               # Retrieval only (chunker, embedders, stores)
CrewAI/            # Agents, tasks, crew, prompts, tools, tracing
api/               # FastAPI routes + AnalysisService
evaluation/        # Classifier metrics
feedback/          # Clinician feedback store + retrain loop
risk/              # Risk history store, trends, escalation alerts
scripts/           # Experiment runners (M2 / M3 / privacy)
data/hospitals/    # Per-hospital local datasets (never committed raw PHI)
artifacts/         # Model artifacts, SQLite registries, experiment reports

frontend/
dashboard/         # Dashboard client package (httpx API client)
streamlit_app.py   # Streamlit doctor dashboard entry point

n8n/               # Workflow JSON exports (source of truth)

docs/              # Architecture decisions, baselines, privacy notes
.ai/               # Session context, backlog, notes
```

Each module is independent and should not contain unrelated business logic.

---

# Module Responsibilities

## preprocessing/

Responsible for data validation and transformation.

Responsibilities

- CSV validation
- Image validation
- Missing value handling
- Encoding
- Scaling
- Feature engineering
- Image normalization
- Data preprocessing pipelines

This module does not perform prediction.

---

## models/

Responsible for machine learning inference.

Responsibilities

- CSV prediction models (`TabularClassifier`, `TorchMLPClassifier`)
- Medical image models (`ImageClassifier` — torch CNN)
- Model loading and lazy loading
- Model inference
- `partial_fit` for incremental training (required by federated clients)
- `get_parameters` / `set_parameters` for weight exchange

This module does not perform preprocessing.

---

## federated/

Responsible for federated learning.

Responsibilities

- Flower server (`FedAvgServer`) and clients (`FederatedClient`)
- Distributed gRPC runs (`distributed.py`) with TLS support
- Per-hospital schema adapters (`canonical.py`) mapping specialty CSVs
  onto a shared canonical feature space
- Local training with optional DP-SGD via Opacus (`privacy.py`)
- Pairwise one-time-pad secure aggregation
- Anonymization/pseudonymization at data loading
- Payload inspection for leakage measurement
- Privacy metrics: per-round ε, cumulative ε upper bound, MIA AUROC,
  attack resistance, data leakage rate
- Model registry (`registry.py`) persisting runs and artifacts

Two federation modes:

1. **Single-preset partitioned** — one preset CSV split across N hospitals.
2. **Heterogeneous** — each hospital trains on its own specialty dataset
   (Diabetes A, Heart B, CKD C, Sepsis D); columns mapped to canonical
   schema; local files never overwritten.

No raw patient data leaves local hospitals. Only model parameters travel.

---

## rag/

Responsible for knowledge retrieval.

Responsibilities

- Document ingestion (static corpus + live source fetchers)
- Embedding generation (TF-IDF default; HashingEmbedder fallback;
  swappable via `Embedder` ABC)
- Vector search (in-memory default; ChromaDB optional extra;
  swappable via `VectorStore` ABC)
- Context retrieval
- RAG evaluation set (18 ground-truth clinical queries)

Knowledge sources include

- KDIGO CKD Guidelines
- Surviving Sepsis Campaign
- ADA Standards of Care
- ACC/AHA Heart Failure
- NICE CKD NG203
- WHO reports
- CDC guidance
- Landmark trial evidence (DAPA-CKD, PARADIGM-HF, EMPA-KIDNEY, …)
- Hospital protocols

---

## CrewAI/

Responsible for multi-agent reasoning.

Agents include

1. Patient Analyst
2. Disease Predictor
3. Medical Researcher (RAG)
4. Treatment Planner
5. Explainability Expert
6. Risk Monitor
7. Report Writer

Responsibilities

- Deterministic tool pipeline by default (`run_analysis()`) so the crew
  works without an LLM key
- Optional LLM layer via NVIDIA NIM or any OpenAI-compatible endpoint
  (`run_llm()`), merged over the deterministic base report
- Full per-agent execution tracing (`AgentTrace` / `CrewTrace`) recording
  input, output, status, and timing for each step
- Agent metrics computed from real traces (task completion rate,
  decision consistency, collaboration score)

CrewAI consumes prediction results but never trains models. Agents never implement ML algorithms.

---

## api/

Responsible for exposing REST APIs.

Responsibilities

- Request validation (Pydantic schemas)
- Optional bearer-token authentication (`API_TOKEN`)
- Response serialization
- Typed error mapping via `APIError` subclasses
- Per-agent orchestration endpoints for n8n step-by-step workflows:
  - `POST /api/v1/agents/patient-analyst`
  - `POST /api/v1/agents/disease-predictor`
  - `POST /api/v1/agents/evidence-retrieval`
  - `POST /api/v1/agents/treatment-planner`
  - `POST /api/v1/agents/explainability`

Business logic remains in the service layer (`AnalysisService`). Routes never contain business logic.

---

## evaluation/

Responsible for benchmarking.

Metrics include

- Accuracy, Precision, Recall, F1, ROC-AUC
- Federated metrics (per-round accuracy deltas, communication cost,
  convergence round)
- Privacy metrics (ε, MIA AUROC, leakage rate, payload inspection)
- RAG metrics (context precision/recall, faithfulness, answer relevancy)
- Agent metrics (task completion, consistency, collaboration)

---

## feedback/

Responsible for clinician feedback collection and retraining.

Responsibilities

- Feedback persistence to SQLite (`artifacts/feedback.db`)
- Pending-feedback counting
- Threshold-gated retraining trigger (n8n polls `/api/v1/feedback/status`)
- Retrain loop on augmented dataset (base + labeled feedback rows)
- Immediate model replacement in memory (no restart needed)

---

## risk/

Responsible for longitudinal risk monitoring.

Responsibilities

- Risk history persistence to SQLite (`artifacts/risk_history.db`)
- Trend analysis per patient across visits
- Escalation alert generation on score jumps above threshold
- Alerts endpoint polled by n8n every 15 minutes

---

# Data Flow

```
CSV / Image
      │
      ▼
Preprocessing (anonymized)
      │
      ▼
Prediction Models
      │
      ▼
CrewAI (7 traced agents)
      │
      ├──▶ RAG evidence retrieval
      │
      ▼
Clinical Report
      │
      ├──▶ Risk history persisted
      │
      ▼
FastAPI response
      │
      ├──▶ n8n workflow (step-by-step agent calls)
      │
      ▼
Storage • Doctor Notify • Dashboard
```

---

# Federated Learning Workflow

```
Hospital A (Pima Diabetes)
Hospital B (UCI Heart Disease)
Hospital C (Chronic Kidney)
Hospital D (MIMIC-IV Sepsis)
      │
      ▼
Canonical Schema Mapping (11 features)
      │
      ▼
Local DP-SGD Training (optional Opacus noise)
      │
      ▼
Pairwise OTP Secure Aggregation
      │
      ▼
Flower Server (FedAvg)
      │
      ▼
Global Model Registry
```

Only model parameters are exchanged. Raw patient data remains within each hospital. Anonymization runs before any frame is used locally. Payload inspection verifies no raw PHI leaks into federation traffic.

---

# Design Principles

- Modular architecture
- Separation of concerns
- Dependency injection
- Reusable preprocessing
- Type safety
- Configuration through environment variables
- Single responsibility per module
- Production-ready code quality
- Reproducible research (fixed seeds, no hidden randomness)
- Measured-over-assumed privacy claims

---

# Coding Rules

- Never duplicate preprocessing logic.
- Never place business logic inside API routes.
- Never hardcode configuration values.
- Always use centralized logging (`get_logger(__name__)`).
- Always use type hints.
- Use Pydantic models for data validation.
- Every public function should include documentation.
- Never log sensitive patient information.
- Never commit raw PHI or credentials.

See `AGENTS.md` for the authoritative list.

---

# Development Workflow

Each feature should follow this sequence:

1. Read `.ai/current_context.md`.
2. Implement the task.
3. Run `ruff check` and `ruff format`.
4. Verify with import smoke test + manual API calls (see README).
5. Update `.ai/current_context.md` and `.ai/next_session.md`.
6. Update relevant docs if the change affects them.
7. Commit with a focused message.

---

# Future Extensions

- Shared-scaler study across heterogeneous hospitals
- Encrypted gRPC cross-host demonstration
- Medical image model training on real datasets
- Tighter ε composition (RDP accountant state persistence)
- Dense sentence-transformer embedder swap-in
- Qdrant persistent vector store backend
- OAuth-based API authentication
