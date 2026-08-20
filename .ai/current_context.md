# Current Context

## Current Milestone

Phase 2 of the multi-hospital federated build: expose the federation
model registry through the FastAPI so the dashboard and external tooling
can inspect distributed runs, versioned global models, and per-round
metrics.

## Current Module

`backend/api/` (schemas, routes, services) reading `federated/registry.py`.

## Current Task

Phase 2 — federation registry API:

1. Schemas: `FederationStatus`, `FederationRun`, `FederationModel`,
   `FederationRound`, `FederationPreset`.
2. Service: `federation_status` / `federation_runs` /
   `federation_models` / `federation_rounds` (registry at
   `<artifacts_dir>/federation.db`).
3. Routes: `GET /api/v1/federation/status|runs|models`,
   `GET /api/v1/federation/runs/{run_id}/rounds`.
4. Stop hardcoding the gRPC address in `_train_distributed` (use
   `FED_SERVER_ADDRESS`).
5. README + `.ai/` updated.

## Completed

- Phase 1 (committed `ebad74f` + `51d479d`): distributed Flower gRPC
  deployment, hospitals-as-processes, SQLite registry, DP + secure
  aggregation, canonical feature schema.
- Federation API endpoints implemented and verified against the live
  server (`diabetes` distributed run, secagg enabled):
  - `/federation/status` → registry path, run/model counts, per-preset
    latest model (with secagg/DP flags from the producing run);
  - `/federation/runs` → run metadata;
  - `/federation/models` → versioned global models;
  - `/federation/runs/{run_id}/rounds` → per-round accuracy / loss.
- `_train_distributed` now reads `FED_SERVER_ADDRESS` instead of
  hardcoding `127.0.0.1:8080`; verified via the API service.
- Live API restarted with the new code; registry created and populated
  through `POST /api/v1/train` with `distributed: true`.

## Next Files (optional / backlog)

- Phase 2+ (backlog): real medical RAG corpora, doctor notification,
  feedback loop, encrypted transport, risk monitoring on history,
  dashboard federation panel.

## Design Notes

- The registry lives at `<artifacts_dir>/federation.db` — the same path
  `_train_distributed` writes via `FED_REGISTRY_PATH`. Service methods
  return empty results when the database does not exist yet (no error).
- `FederationModel.secure_aggregation` / `differential_privacy` are
  filled only by `/federation/status` (joined from the run); the flat
  `/federation/models` list leaves them null.
- Routes remain thin; all registry logic is in the service.

## Status

Phase 2 federation registry API complete and verified against the live
server. Repo not yet committed (user drives commits).