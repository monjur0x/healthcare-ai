# Current Context

## Current Milestone

Phase 2 of the multi-hospital federated build: surface the federation
model registry through the FastAPI and the Streamlit dashboard so
clinicians and operators can inspect distributed runs, versioned global
models, and per-round metrics — and trigger new distributed training.

## Current Module

`frontend/streamlit_app.py` + `frontend/dashboard/client.py` (Federation
tab), reading `backend/api/v1/federation/*`.

## Current Task

Phase 2 — dashboard federation panel (complete):

1. `HealthcareAPIClient` federation methods: `federation_status`,
   `federation_runs`, `federation_models`, `federation_rounds`,
   `train_distributed` (all `distributed` / DP / secagg knobs).
2. Federation tab in `streamlit_app.py`:
   - Registry overview (run/model counts + registry path);
   - Per-condition latest models (version, accuracy, AUC, DP epsilon,
     secagg / DP flags);
   - Distributed training trigger (condition, clients, rounds, secagg,
     DP-SGD) → run result summary;
   - Recent runs selector + per-round accuracy chart.
3. Tab wired into `main()` as the sixth page; module docstring updated.
4. README + `.ai/` updated.

## Completed

- Phase 1 (committed `ebad74f` + `51d479d`): distributed Flower gRPC
  deployment, hospitals-as-processes, SQLite registry, DP + secure
  aggregation, canonical feature schema.
- Federation API endpoints (committed `7afa18e`): `/federation/status`,
  `/federation/runs`, `/federation/models`,
  `/federation/runs/{run_id}/rounds`.
- `_train_distributed` reads `FED_SERVER_ADDRESS` (no hardcoded address).
- Dashboard Federation tab implemented and verified:
  - `streamlit.testing.v1.AppTest`: 0 exceptions on the Federation tab;
    metrics, models dataframe, run selector, and per-round chart render
    against the live backend (run `ee3d29b91b44`, secagg on).
  - Live Streamlit restarted at `127.0.0.1:8501` with the new panel.

## Next Files (optional / backlog)

- Real medical RAG corpora.
- Doctor notification via n8n.
- Feedback / retrain loop.
- Encrypted gRPC transport.
- Risk monitoring on history.
- Cross-host deployment docs.

## Design Notes

- The registry lives at `<artifacts_dir>/federation.db` — the same path
  `_train_distributed` writes via `FED_REGISTRY_PATH`. Service methods
  return empty results when the database does not exist yet (no error).
- The dashboard falls back gracefully: when the backend does not expose
  the federation endpoints, the tab shows an info message instead of
  raising.
- The tab reuses `ASSESSMENT_LABELS` for doctor-friendly condition names
  and `PRESET_PRESETS` as the empty-registry fallback.
- `train_distributed` only sends DP-SGD hyperparameters when
  `differential_privacy` is enabled (matches the backend model).

## Status

Phase 2 federation registry API and dashboard panel complete and verified
against the live server. Repo not yet committed (user drives commits).