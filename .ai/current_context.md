# Current Context

## Current Milestone

Milestone 2 — Models (in progress)

## Current Module

backend/federated (tie-in complete)

## Current Task

Federated tie-in is complete. Remaining scoped items: a full flwr
server/simulation driver, federating the CNN end-to-end (needs a torch
`partial_fit`), and consuming `CSVPipeline` output directly.

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
- `flwr>=1.33.0` added to `backend/requirements.txt` (installed in
  CrewAI venv)
- Tests: models 32, evaluation 11, federated 18; full suite 131 passing

## Next Files (backend)

- `federated/` — flwr `ServerApp` / FedAvg strategy driver +
  simulation; federated metrics (communication cost / convergence /
  training time)
- CNN federation — add torch `partial_fit` (continue from current
  weights) so the image path can join rounds
- consume `CSVPipeline` output directly (accepts DataFrame today)

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

Milestone 2 (models + evaluation + federated tie-in) substantially
complete. Remaining: server driver, CNN federation, direct CSV pipeline
consumption.