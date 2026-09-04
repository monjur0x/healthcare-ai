# Archived n8n workflows (retired, kept for reference)

These workflows are superseded and no longer part of the documented
setup. They remain importable, but new deployments should use the
canonical set in `n8n/`:

- `clinical-analysis.json` (`POST /webhook/healthcare-analyze`) —
  analyze-only minimal flow. Superseded by `healthcare-endtoend.json`,
  which already supports analyze-only requests.
- `clinical-full.json` (`POST /webhook/clinical-full`) — proposal
  10-step orchestration through the monolithic analyze call.
  Superseded by `clinical-full-v2.json`
  (`POST /webhook/clinical-full-v2`), which routes the same 10 steps
  through the per-agent `/api/v1/agents/*` endpoints.

Canonical set: `healthcare-endtoend.json`, `clinical-full-v2.json`,
`risk-monitoring.json`, `feedback-retrain.json`.
