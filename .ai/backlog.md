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
  summaries).
- [x] DONE: §12 RAG metrics wired via scripts/run_m3_evaluation.py;
  FAITHFULNESS_THRESHOLD now configurable (RAG_FAITHFULNESS_THRESHOLD) —
  set ~0.3 for TF-IDF (ceiling 0.66 verified) or use dense embedder at
  default 0.5.

## n8n

- [x] DONE (M3.4): `n8n/clinical-full.json` implements the 10-step flowchart
  workflow; verified rejection / low-risk / high-risk-notify paths live.
- [x] DONE: `n8n/risk-monitoring.json` polls `/api/v1/risk/alerts`
  every 15m and notifies clinicians per alert (activated locally).

## Dashboard

- [x] DONE: new "Risk Monitoring" tab — per-patient trend chart +
  direction metric, active escalation-alert list, clinician feedback
  form posting to `/api/v1/feedback` (client methods added).

## Housekeeping

- Image model training pipeline (currently inference-only; proposal
  future-extension §15 lists medical-image integration).
