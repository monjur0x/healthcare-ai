# AGENTS.md

# AI Development Guide

## Project

**Federated Multi-Agent Healthcare Intelligence Framework**

This repository implements a research-oriented healthcare AI framework integrating:

- Federated Learning (Flower)
- Multi-Agent Systems (CrewAI)
- Retrieval-Augmented Generation (RAG)
- Medical Image Analysis
- Electronic Health Record (CSV) Analysis
- FastAPI
- n8n Orchestration
- Next.js Dashboard

The goal is to produce a clean, modular, production-quality implementation suitable for academic research.

---

# Read Before Coding

Before making ANY code changes, read these files in order:

1. docs/SYSTEM_SPECIFICATION.md
2. docs/SOFTWARE_ARCHITECTURE.md
3. docs/DEVELOPMENT_STATUS.md
4. docs/BACKLOG.md
5. docs/DECISIONS.md
6. .ai/current_context.md

Never skip this step.

---

# Repository Structure

backend/
    preprocessing/
    models/
    federated/
    rag/
    CrewAI/
    api/
    evaluation/

frontend/

n8n/

docs/

.ai/

---

# Task Execution Rules

When starting work:

1. Read `.ai/current_context.md`
2. Identify the current task.
3. Implement ONLY that task.
4. Do not start unrelated refactoring.
5. Keep changes focused.
6. If additional work is discovered, record it in `docs/BACKLOG.md` instead of implementing it immediately.
7. Before finishing, update:
   - `.ai/current_context.md`
   - `.ai/next_session.md`
   - `docs/DEVELOPMENT_STATUS.md`
   - `docs/CHANGELOG.md`

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

3.12+

Formatting

- Black
- Ruff
- isort

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

Every new module should include tests whenever practical.

Tests belong in:

tests/

Do not reduce existing test coverage.

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

Never perform preprocessing inside Flower clients.

Use preprocessing outputs.

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

Run tests

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

docs/DEVELOPMENT_STATUS.md

docs/CHANGELOG.md

docs/BACKLOG.md

.ai/current_context.md

.ai/next_session.md

If an architectural decision changed:

Update:

docs/DECISIONS.md

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