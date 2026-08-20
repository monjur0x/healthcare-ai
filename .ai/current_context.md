# Current Context

## Current Milestone

Phase 1 of the multi-hospital federated build: a genuine distributed
Flower deployment where each hospital runs as its own process and
exchanges only model weights with a real Flower gRPC server, with an
SQLite model registry for versioned global models.

## Current Module

`backend/federated/` (hospitals, distributed, registry, launcher) plus
`backend/preprocessing/loader.py` (shared CSV loader).

## Current Task

Phase 1 — multi-hospital federated, verified end to end:

1. `preprocessing/loader.py` — canonical `load_classification_frame`
   (CSV → engineered features + encoded labels); API and hospitals share it.
2. `federated/hospitals.py` — `HospitalConfig` + `build_hospital_sites`
   (partitions a preset into per-hospital local CSVs) + `load_hospital_dataset`.
3. `federated/distributed.py` — `DistributedFedAvg` (FedAvg strategy
   keeping the pairwise OTP secure-aggregation semantics), `ModelSpec`,
   `run_distributed_server`, `run_hospital_client` (Flower gRPC).
4. `federated/registry.py` — SQLite `ModelRegistry` (runs, rounds, models).
5. `federated/__main__.py` — `python -m federated` CLI: `run`,
   `server`, `client`, `sites`.
6. API — `TrainRequest.distributed` flag → `services._train_distributed`
   runs the distributed deployment and reports registry metrics.
7. README + `.ai/` updated.

## Completed

- Flower 1.33 API verified: `start_server` / `start_numpy_client` over
  gRPC work; no built-in SecureAggregation strategy (custom strategy used).
- Hospital data layer: 4-site partition + central hold-out slice verified.
- `DistributedFedAvg` masked (secure) aggregation + weighted (plain)
  aggregation verified; per-round metrics persist to SQLite.
- Registry verified: runs, rounds, models, `latest_model`, versioning.
- CLI `run` verified end to end for all four presets (diabetes, heart,
  kidney, sepsis) with plain, secure-aggregation, and DP+secagg variants.
- API distributed train verified end to end (`preset=heart`, `kidney`):
  returns `federated_metrics` with run_id/version.
- DP + secure aggregation path verified end to end (epsilon recorded).
- Fixed per-slice feature drift: model spec is now derived from the full
  source CSV (`_spec_from_preset`), and the server-side hold-out
  evaluation falls back to client-aggregated accuracy when the hold-out
  slice preprocesses to a different feature count (heart: 14 vs 11,
  kidney: 19 vs 22 due to all-NaN columns in the contiguous slice).
- Fixed string-target hospital partitioning (`LabelEncoder` for
  `classification` values like "ckd" instead of `.astype(int)`).
- Ruff clean; format applied; all imports compile; live API/dashboard/n8n
  healthy.

## Next Files (optional / backlog)

- `.ai/next_session.md` update.
- Phase 2+ (backlog): real medical RAG corpora, doctor notification,
  feedback loop, encrypted transport, risk monitoring on history.

## Design Notes

- Hospital slice partitioning uses `StratifiedKFold` (rarest class must
  support n_sites). Each hospital preprocesses its own CSV locally.
- Model spec (`n_features`/`n_classes`) is derived from the **full**
  source CSV, not a single slice, so all clients agree on the
  architecture even when slices preprocess to different shapes.
- Server-side hold-out evaluation is optional: if the hold-out slice
  preprocesses to a different feature count than the model, it logs a
  warning and uses the client-aggregated accuracy instead (avoids
  crashing the whole run on a contiguous all-NaN column slice).
- `_train_distributed` spawns the `federated run` launcher as a
  subprocess; registry + artifacts are written under `API_ARTIFACTS_DIR`.

## Status

Phase 1 complete and verified across all four presets. Repo not yet
committed (user drives commits).