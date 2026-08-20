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

Phase 1 follow-up — resolve per-slice feature drift (hold-out and client
alignment):

1. `federated/hospitals.py` — stratified, class-balanced, disjoint
   central hold-out (one fold of an `n_sites+1` split) instead of a
   contiguous slice.
2. `federated/distributed.py` — `ModelSpec.feature_names` canonical
   schema + `align_features()` so every participant reindexes/zero-fills
   its local features to the full-dataset columns.
3. `federated/__main__.py` — CLI passes the feature schema to server and
   client subprocesses.
4. Re-verify all four presets through CLI and the API service.

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
- **Fixed per-slice feature drift** (contiguous hold-out was class-sorted
  and lost all-NaN columns → shape mismatch). Now:
  - the hold-out is a stratified, disjoint fold of the full data;
  - `ModelSpec` carries the canonical `feature_names` and every client /
    the server align their local features to that schema;
  - hold-out evaluation now succeeds for heart (acc 0.54, AUC 0.73),
    kidney (acc 0.87), sepsis (acc 1.0) — no more fallback.
- Fixed string-target hospital partitioning (`LabelEncoder` for
  `classification` values like "ckd" instead of `.astype(int)`).
- Ruff clean; format applied; all imports compile; live API/dashboard/n8n
  healthy.

## Next Files (optional / backlog)

- `.ai/next_session.md` update.
- Phase 2+ (backlog): real medical RAG corpora, doctor notification,
  feedback loop, encrypted transport, risk monitoring on history.

## Design Notes

- Hospital slice partitioning uses `StratifiedKFold` with `n_sites + 1`
  folds: one fold becomes the central hold-out, the rest are the
  hospital sites. Everything is class-balanced, disjoint, and
  column-complete.
- The model spec (`n_features`/`n_classes`/`feature_names`) is derived
  from the **full** source CSV. Each participant aligns its local
  features via `ModelSpec.align_features()` (reindex + zero-fill), so
  per-slice imputer behavior cannot change the matrix shape.
- `_train_distributed` spawns the `federated run` launcher as a
  subprocess; registry + artifacts are written under `API_ARTIFACTS_DIR`.

## Status

Phase 1 complete and feature-drift fix verified across all four presets.
Repo not yet committed (user drives commits).