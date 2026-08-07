# Software Architecture

## Overview

The Healthcare AI Framework is a modular, service-oriented system for privacy-preserving clinical decision support. The system processes structured healthcare data (CSV) and medical images, performs AI inference using machine learning models, coordinates reasoning through CrewAI agents, retrieves medical evidence using RAG, and automates clinical workflows using n8n.

The architecture is designed around separation of concerns. Each module has a single responsibility and communicates through well-defined interfaces.

---

# High-Level Architecture

```
                    User
                      │
                      ▼
              Next.js Frontend
                      │
                      ▼
               FastAPI Backend
                      │
      ┌───────────────┴────────────────┐
      │                                │
      ▼                                ▼
 CSV Preprocessing             Image Preprocessing
      │                                │
      └───────────────┬────────────────┘
                      ▼
              Prediction Models
                      │
                      ▼
            CrewAI Multi-Agent System
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
              n8n Orchestration
                      │
                      ▼
         Storage • Notification • Dashboard
```

---

# Backend Structure

```
backend/

preprocessing/
models/
CrewAI/
rag/
federated/
api/
evaluation/
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

- CSV prediction models
- Medical image models
- Multimodal fusion
- Model loading
- Model inference

This module does not perform preprocessing.

---

## CrewAI/

Responsible for multi-agent reasoning.

Agents include

- Patient Analysis Agent
- Disease Prediction Agent
- RAG Agent
- Treatment Recommendation Agent
- Explainability Agent
- Risk Monitoring Agent
- Report Generation Agent

CrewAI consumes prediction results but never trains models.

---

## rag/

Responsible for knowledge retrieval.

Responsibilities

- Document ingestion
- Embedding generation
- Vector search
- Context retrieval

Knowledge sources include

- PubMed
- WHO
- CDC
- NICE
- Hospital protocols

---

## federated/

Responsible for federated learning.

Responsibilities

- Flower server
- Flower clients
- Federated aggregation
- Local training
- Global model synchronization

No raw patient data leaves local hospitals.

---

## api/

Responsible for exposing REST APIs.

Responsibilities

- Request validation
- Authentication
- File upload
- Response serialization

Business logic should remain in service modules.

---

## evaluation/

Responsible for benchmarking.

Metrics include

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- Privacy metrics
- Federated metrics
- RAG metrics

---

# Data Flow

```
CSV / Image
      │
      ▼
Preprocessing
      │
      ▼
Prediction Models
      │
      ▼
CrewAI
      │
      ▼
RAG
      │
      ▼
Clinical Report
      │
      ▼
FastAPI
      │
      ▼
n8n
```

---

# Federated Learning Workflow

```
Hospital A
Hospital B
Hospital C
Hospital D
      │
      ▼
Local Training
      │
      ▼
Flower Server
      │
      ▼
FedAvg
      │
      ▼
Global Model
```

Only model parameters are exchanged. Raw patient data remains within each hospital.

---

# Design Principles

The project follows these principles:

- Modular architecture
- Separation of concerns
- Dependency injection
- Reusable preprocessing
- Type safety
- Configuration through environment variables
- Single responsibility per module
- Production-ready code quality

---

# Coding Rules

- Never duplicate preprocessing logic.
- Never place business logic inside API routes.
- Never hardcode configuration values.
- Always use centralized logging.
- Always use type hints.
- Use Pydantic models for data validation.
- Every public function should include documentation.

---

# Development Workflow

Each feature should follow this sequence:

1. Implement the feature.
2. Add unit tests.
3. Run formatting and linting.
4. Update project documentation.
5. Update development status.
6. Commit changes.

---

# Future Extensions

The architecture is designed to support additional healthcare datasets, new prediction models, additional CrewAI agents, and new RAG knowledge sources without requiring major structural changes.