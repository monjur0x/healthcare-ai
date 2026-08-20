# Next Session

## Objective

Continue the multi-hospital federated build (Phase 1 is complete and
verified). Next: decide on commit cadence and Phase 2 scope.

## Done This Session

- Committed + pushed Phase 1 (`ebad74f`).
- Fixed per-slice feature drift at the root:
  - hold-out is now a stratified, class-balanced, disjoint fold
    (`StratifiedKFold(n_sites+1)`), not a contiguous slice;
  - `ModelSpec` gained a canonical `feature_names` schema derived from
    the full dataset, and `align_features()` reindexes/zero-fills every
    participant's local features;
  - CLI passes `--feature-names` to server and client subprocesses.
- Hold-out evaluation now succeeds for all presets (no fallback):
  heart acc 0.54 / AUC 0.73, kidney acc 0.87, sepsis acc 1.0, via CLI
  and via the API service (`federated_metrics.roc_auc` populated).
- Updated README + `.ai/current_context.md`.

## Next Steps

1. Commit the feature-drift fix when the user asks.
2. Phase 2 candidates (backlog):
   - Real medical RAG corpora (replace the built-in 3-text corpus).
   - Doctor notification (n8n alert when risk is high).
   - Feedback loop (n8n → retrain → redeploy on threshold).
   - Encrypted gRPC transport (Flower certificates).
   - Risk monitoring over persisted patient history.
   - Cross-host deployment of hospital clients (documented commands).
3. Consider exposing registry models (`GET /api/v1/federation/models`).

## Open Questions

- Commit cadence for the feature-drift fix.
- Whether the API `_train_distributed` should stop hardcoding the gRPC
  address (currently 127.0.0.1:8080) and read from `FED_` settings.