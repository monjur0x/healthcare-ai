# Next Session

## Objective

Continue the multi-hospital federated build (Phase 1 is complete and
verified). Next: decide on commit cadence and Phase 2 scope.

## Done This Session

- Added shared CSV loader `preprocessing/loader.py`
  (`load_classification_frame`) used by both the API and the hospital
  data layer.
- Added `federated/hospitals.py` — `HospitalConfig`, `build_hospital_sites`
  (per-hospital local CSV slices + central hold-out), `load_hospital_dataset`.
- Added `federated/distributed.py` — `ModelSpec`, `DistributedFedAvg`
  strategy (pairwise OTP secure aggregation), `run_distributed_server`,
  `run_hospital_client` over Flower gRPC.
- Added `federated/registry.py` — SQLite `ModelRegistry` (runs, per-round
  metrics, versioned global models).
- Added `federated/__main__.py` — `python -m federated` CLI (`run`,
  `server`, `client`, `sites`).
- Wired `distributed` flag through `api/schemas.py`, `api/routes.py`,
  and `api/services.py::_train_distributed`.
- Fixed string-target hospital partitioning (kidney `classification`).
- Fixed per-slice feature drift: spec now derived from the full dataset;
  hold-out evaluation falls back to client-aggregated accuracy on shape
  mismatch (heart, kidney).
- Verified all four presets end to end (plain / secagg / DP+secagg) via
  CLI and via the API service.
- Updated `README.md` (distributed federation section) and
  `.ai/current_context.md`.

## Next Steps

1. Commit Phase 1 when the user asks (keep it one focused commit).
2. Phase 2 candidates (backlog):
   - Real medical RAG corpora (replace the built-in 3-text corpus).
   - Doctor notification (n8n alert when risk is high).
   - Feedback loop (n8n → retrain → redeploy on threshold).
   - Encrypted gRPC transport (Flower certificates).
   - Risk monitoring over persisted patient history.
   - Cross-host deployment of hospital clients (documented commands).
3. Consider exposing registry models (`GET /api/v1/federation/models`).
4. Consider whether the hold-out slice should be stratified/consistent
   with the sites (currently contiguous, which causes feature drift).

## Open Questions

- Whether the per-preset feature drift on hold-out evaluation should be
  resolved (currently falls back to client-aggregated accuracy).
- Commit cadence for the remaining Phase 1 files.