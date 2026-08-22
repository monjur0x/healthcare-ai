# AGENTS.md

# AI Development Guide

## Project

**Federated Multi-Agent Healthcare Intelligence Framework**

This repository implements a research-oriented healthcare AI framework integrating:

- Federated Learning (Flower) — partitioned + heterogeneous multi-disease
- Multi-Agent Systems (CrewAI)
- Retrieval-Augmented Generation (RAG)
- Medical Image Analysis
- Electronic Health Record (CSV) Analysis
- Clinician Feedback / Retrain Loop
- Longitudinal Risk Monitoring & Escalation Alerts
- FastAPI
- n8n Orchestration
- Streamlit Dashboard

The goal is to produce a clean, modular, production-quality implementation suitable for academic research.

---

# Read Before Coding

Before making ANY code changes, read these files in order:

1. README.md
2. .ai/current_context.md

Never skip this step.

---

# Repository Structure

backend/
    preprocessing/        # CSV / image pipelines, canonical loaders (no ML)
    models/               # ML/DL models only (tabular MLP, torch MLP, image CNN)
    federated/            # Flower FL: server, clients, hospitals, canonical schema
    rag/                  # Retrieval only (chunker, embedders, stores)
    CrewAI/               # Agents, tasks, crew, prompts, tools
    api/                  # FastAPI routes + AnalysisService
    evaluation/           # Classifier metrics
    feedback/             # Clinician feedback store + retrain loop
    risk/                 # Risk history store, trends, escalation alerts
    scripts/              # Experiment runners (e.g. run_m2_experiment.py)
    data/hospitals/       # Per-hospital local datasets (NEVER committed raw PHI)
    artifacts/            # Model artifacts, SQLite registries, experiment reports

frontend/
    dashboard/            # Dashboard client package
    streamlit_app.py      # Doctor dashboard entry point

n8n/                      # Workflow JSON exports (source of truth)

.ai/                      # Session context, backlog, notes

scripts/                  # Repo-level helpers (demo launcher)

---

# Multi-Hospital Data Layout (per research proposal)

Each hospital owns a DIFFERENT specialty dataset; no hospital shares raw
data:

- Hospital A — Pima Diabetes        (`data/hospitals/hospital_A/data.csv`)
- Hospital B — UCI Heart Disease    (`data/hospitals/hospital_B/data.csv`)
- Hospital C — Chronic Kidney       (`data/hospitals/hospital_C/data.csv`)
- Hospital D — MIMIC-IV-style sepsis(`data/hospitals/hospital_D/data.csv`)

Two federation modes exist:

1. **Single-preset (partitioned)** — one preset CSV split across N
   hospitals via `build_hospital_sites`. Legacy/simulation mode.
2. **Heterogeneous (`--heterogeneous`)** — each hospital trains on its
   own specialty CSV as-is. Local files are NEVER overwritten or
   repartitioned in this mode.

FedAvg across different diseases works because every hospital maps its
columns onto the shared canonical schema (`federated/canonical.py`,
derived from the proposal's "Expected Inputs") with a binary
`has_disease` target. Missing canonical features are zero-filled by
`ModelSpec.align_features`.

---

# Task Execution Rules

When starting work:

1. Read `.ai/current_context.md`
2. Identify the current task.
3. Implement ONLY that task.
4. Do not start unrelated refactoring.
5. Keep changes focused.
6. If additional work is discovered, record it in `.ai/backlog.md` instead of implementing it immediately.
7. Before finishing, update:
   - `.ai/current_context.md`
   - `.ai/next_session.md`

---

# Development Principles

Always prefer:

- Clean Architecture
- SOLID Principles
- DRY
- KISS
- Separation of Concerns

Every module must have a single responsibility.

Never place business logic inside API routes.

Never duplicate code.

Never hardcode values.

Configuration belongs in `.env` or configuration classes.

---

# Coding Standards

Python Version

3.11+ (validated on 3.13/3.14; crewai and chromadb are optional
extras — see requirements.txt)

Formatting

- Ruff (`ruff check`, `ruff format`) — the only formatters/linters used

Typing

Every public function must use type hints.

Avoid `Any` unless absolutely necessary.

Use dataclasses or Pydantic models whenever appropriate.

---

# Logging

Never use:

print()

Always use:

logger = get_logger(__name__)

Every important action should be logged.

---

# Error Handling

Never silently ignore exceptions.

Raise custom exceptions whenever possible.

Catch exceptions only where they can be handled meaningfully.

---

# Documentation

Every public:

- Class
- Function
- Method

must contain docstrings.

Complex algorithms should include explanatory comments.

---

# Testing

Tests were removed from the repository by request. The live system is
verified by the import smoke check and manual API calls described in
README.md.

---

# Architecture Rules

## Preprocessing

Only preprocessing logic.

No machine learning.

No API logic.

No CrewAI.

---

## Models

Only ML/DL models.

No API logic.

No RAG.

---

## Federated

Only Flower/Federated Learning.

Preprocessing *logic* lives in `preprocessing/` (and the per-hospital
schema adapters in `federated/canonical.py`); never duplicate it inside
client/server code.

Preprocessing *execution* happens locally at each hospital on its own
CSV — raw rows never leave a hospital; only model weights travel.

---

## RAG

Only retrieval logic.

No prediction logic.

No API logic.

---

## CrewAI

Contains:

- Agents
- Tasks
- Crew
- Prompt Templates
- Tools

Agents should orchestrate reasoning.

Agents should NOT implement ML algorithms.

Agents consume outputs from preprocessing and prediction models.

---

## API

FastAPI routes only.

Business logic belongs in services.

---

## n8n

Contains orchestration only.

Never move AI reasoning into n8n.

n8n triggers workflows.

CrewAI performs reasoning.

---

# Research Constraints

This repository is intended for academic research.

Maintain reproducibility.

Avoid introducing hidden randomness.

Use fixed random seeds where appropriate.

Record important architectural decisions.

---

# Performance

Avoid unnecessary memory copies.

Use vectorized operations.

Prefer batch processing.

Lazy-load large models whenever possible.

---

# Security

Never commit:

.env

API Keys

Tokens

Passwords

Private datasets

Hospital records

Generated credentials

Never log sensitive patient information.

---

# Dependency Rules

Before adding a dependency:

1. Check if an existing dependency already solves the problem.
2. Keep dependencies minimal.
3. Update requirements if needed.

---

# Git Rules

Keep commits focused.

One logical feature per commit.

Examples

GOOD

feat(preprocessing): add CSV validator

GOOD

feat(rag): implement Qdrant retriever

BAD

misc updates

---

# Workflow

Every coding session should follow this sequence.

Read documentation

↓

Read current context

↓

Implement task

↓

Run formatter

↓

Run linter

↓

Update documentation

↓

Update progress

↓

Prepare next session

---

# After Completing Any Task

Always update:

.ai/current_context.md

.ai/next_session.md

---

# Never Do These

❌ Duplicate preprocessing logic

❌ Hardcode paths

❌ Hardcode API keys

❌ Mix API with business logic

❌ Skip typing

❌ Skip documentation

❌ Skip updating documentation

❌ Delete existing functionality without explanation

---

# If Unsure

If requirements are ambiguous:

1. Follow the existing architecture.
2. Preserve backward compatibility.
3. Leave a clear TODO comment.
4. Record the uncertainty in `.ai/session_notes.md`.

Never invent functionality that contradicts the existing design.

---

# Goal

The objective is not only to produce working code.

The objective is to build:

- maintainable code
- reproducible research
- modular architecture
- production-quality software
- a repository suitable for publication and long-term maintenance.

Every change should improve the project toward that goal.