# Current Context

## Current Milestone

Milestone 4 — CrewAI orchestration (complete)

## Current Module

backend/CrewAI/orchestrator

## Current Task

The orchestrator module and its demo are complete and pushed. Next:
FastAPI (`api/`), then n8n orchestration.

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
- Full suite **222 passing** (`pytest preprocessing/tests models/tests evaluation/tests federated/tests rag/tests examples/tests CrewAI/orchestrator/tests`)
  — black / isort / ruff clean

## Next Files (backend)

- `api/` FastAPI routes (services only; no business logic in routes);
  endpoints to run the clinical crew, predict, and retrieve
- `n8n/` orchestration workflows that trigger the API / crew
- `federated/` — real flwr `run_simulation` / networked `ServerApp`
  (blocked: `ray` not installed); privacy budget metrics
- Orchestrator LLM path: wire a provider (needs `crewai[google-genai]`
  or similar + API key; never commit secrets)

## Design Notes

- ADR-008: the crew runs a deterministic tool pipeline by default
  (prediction → risk → evidence → report) that needs no LLM and is
  fully testable; the CrewAI layer is optional narrative enrichment.
- Agents/tasks/crew construct without an LLM key; `create_agents(llm=...)`
  binds the provider only on the LLM path. CrewAI 1.15 warns internally
  (deprecations) — unrelated to this module.
- `ClinicalCrew` needs `features` (full preprocessed row) when `model`
  is set; `markers` (raw clinical values) feed risk factor flags.
- CrewAI venv (`backend/CrewAI/.venv-opencode`) has crewai 1.15.11,
  pydantic 2.12, qdrant-client, sentence-transformers, flwr, torch.
- Existing `CrewAI/app/*` is the old FastAPI demo; the new module is
  `CrewAI/orchestrator/` (backward compatible).

## Status

Milestone 4 (CrewAI orchestration + demo) complete and pushed. Next
milestone is FastAPI (`api/`), then n8n.