# Backlog

Discovered-but-not-now items. Record here instead of implementing
mid-task (see AGENTS.md → Task Execution Rules).

## Privacy Layer (proposal §8 / flowchart AN node)

- Wire `federated/privacy.anonymize_frame` + `pseudonymize` into the
  hospital data-loading path so PII-like columns are dropped before any
  local training (currently implemented but unwired).

## Federation quality

- Shared-scaler study: clients currently train on unscaled canonical
  features (per-hospital scalers would desync FedAvg weight spaces).
  Evaluate round-0 federated standardization vs raw.
- Encrypted gRPC end-to-end demo across hosts with generated certs.

## Multi-Agent (proposal §6)

- Expose the six-agent CrewAI pipeline (A1 patient analysis … A6 risk
  monitoring) as distinct agents in `run_llm`, keeping the deterministic
  fallback path authoritative for prediction/risk/evidence.
- Wire `CrewAI/orchestrator/metrics.compute_agent_metrics` into the crew
  run and surface agent metrics in the report.

## RAG

- Expand bundled corpus (more conditions, WHO/CDC/NICE-style guideline
  summaries); wire `rag/metrics.retrieval_metrics` /
  `rag_quality_metrics` into an evaluation script per proposal §12.

## n8n

- Single 10-step workflow matching the flowchart (validate → federated
  prediction → RAG → treatment → explainability → store → notify),
  replacing the current three-workflow split where sensible.
- Risk-monitoring workflow polling `/api/v1/risk/alerts` on schedule.

## Dashboard

- Risk-history trend charts and escalation-alert panel.
- Feedback annotation UI feeding `/api/v1/feedback`.

## Housekeeping

- Image model training pipeline (currently inference-only; proposal
  future-extension §15 lists medical-image integration).
