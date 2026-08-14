# Current Context

## Current Milestone

Milestone 2 — Models (in progress)

## Current Module

backend/federated (tie-in complete)

## Current Task

Federated tie-in, CSV → FedAvg demo, CNN federation, and federated
metrics are complete. Remaining scoped items: a full flwr
`run_simulation` / networked `ServerApp` deployment (blocked: `ray`
not installed).

## Completed

- Milestone 1: preprocessing (CSV + image + multimodal), 70 tests
- Shared model interface (`models/base.py`) and exceptions
- `TabularClassifier` — sklearn GB / logistic / MLP, joblib persistence
- `ImageClassifier` — torch CNN, channels-last or channels-first input
- `FusionClassifier` — MLP over `FusionResult.fused`
- `models/config.py` — `ModelSettings` (env prefix `MODEL_`)
- `evaluation/metrics.py` — `ClassificationMetrics`,
  `classification_metrics`, `evaluate_classifier`
- Weight exchange on models: `get_parameters` / `set_parameters`
  (tabular logistic/MLP, fusion, CNN); `partial_fit` (MLP only — sklearn
  LogisticRegression/GB lack incremental training, see ADR-005)
- `federated/parameters.py` — `average_weights` (FedAvg)
- `federated/client.py` — `FederatedClient` (flwr 1.33 `NumPyClient`):
  warm start, one local partial-fit per round, log-loss + accuracy eval
- `federated/server.py` — synchronous `FedAvgServer` driver (initial
  aggregation, per-round client fit → aggregate → evaluate) +
  `make_global_evaluator` (central hold-out scoring); mirrors flwr
  `FedAvg` without the Ray process spawn
- `TabularClassifier.set_parameters` materializes unfitted estimators
  via deterministic dummy fit (structure only)
- `flwr>=1.33.0` added to `backend/requirements.txt` (installed in
  CrewAI venv)
- `backend/examples/fedavg_demo.py` — end-to-end CSV → `CSVPipeline` →
  `TabularClassifier` (MLP) → `FedAvgServer` → evaluation report;
  presets for diabetes / heart / kidney / sepsis; writes
  `global_model.joblib` + `report.json`; `RoundResult.to_dict()` for
  JSON metrics
- `ImageClassifier.partial_fit` — one-epoch incremental CNN training
  from current weights (labels restricted to fit-time classes); the
  image path joins federated rounds via the existing client/server
  (ADR-006). `BaseModel` documents a default `partial_fit`.
- CNN federation end-to-end tests (`federated/tests/test_cnn_federation.py`)
- `federated/metrics.py` — `FederatedMetrics` + `parameter_set_bytes`,
  `round_accuracy_deltas`, `convergence_round`; `FedAvgServer.run()`
  records per-round duration + bytes (client upload + broadcast),
  surfaced via `RoundResult` fields and the `server.metrics` property;
  demo report includes a `federated_metrics` section
- Tests: models 36, evaluation 11, federated 35; full suite 152 passing

## Next Files (backend)

- `federated/` — real flwr `run_simulation` / networked `ServerApp`
  deployment (needs `ray` installed); privacy budget metrics
- Extend `fedavg_demo.py` to the image path using the brain-tumor MRI
  dataset (needs a GPU for practical runtimes)

## Design Notes

- Models expose weight exchange; federation only orchestrates weights.
- `FederatedClient._build()` warm-starts an unfitted factory model on
  local data so weight structure/classes are materialized before
  `set_parameters`; each round rebuilds the model and applies the
  aggregated global weights before one `partial_fit` pass.
- `TabularClassifier.get_parameters` interleaves coefs/intercepts
  (alternating W/b) — `set_parameters` mirrors that order.
- `partial_fit` requires `model_name="mlp"` (ADR-005); logistic/GB
  exchange weights but cannot do incremental local steps.
- Reproducibility: `MODEL_RANDOM_SEED` in `models/config.py`.
- Testing: use the CrewAI venv (`backend/CrewAI/.venv-opencode`) which
  has sklearn, torch, flwr 1.33.0.
- Existing `CrewAI/app/models/*` are old demos; do not mix with
  `backend/models/`.

## Status

Milestone 2 (models + evaluation + federated tie-in + sync server
driver + CSV → FedAvg demo + CNN federation + federated metrics)
substantially complete. Remaining: real flwr simulation/deployment
(blocked on `ray`).