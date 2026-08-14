# Current Context

## Current Milestone

Milestone 2 — Models (in progress)

## Current Module

backend/evaluation

## Current Task

Evaluation hooks are complete. Remaining scoped items: the Flower
federated tie-in (`federated/`) and consuming `CSVPipeline` output
directly.

## Completed

- Milestone 1: preprocessing (CSV + image + multimodal), 70 tests
- Shared model interface (`models/base.py`) and exceptions
- `TabularClassifier` (`models/csv/tabular.py`) — sklearn gradient
  boosting / logistic / mlp, joblib persistence
- `ImageClassifier` (`models/image/cnn.py`) — torch CNN, channels-last
  (N, H, W, C) or channels-first input, adaptive pooling, deterministic
  training, `torch.save` / `load`
- `FusionClassifier` (`models/multimodal/fusion_model.py`) — MLP over
  `FusionResult.fused`, composes `TabularClassifier`
- `models/config.py` — `ModelSettings` (seed, epochs, batch size,
  learning rate, device; env prefix `MODEL_`); tabular seed unified here
- `evaluation/metrics.py` — `ClassificationMetrics`, `classification_metrics`,
  `evaluate_classifier` (accuracy, macro P/R/F1, MCC, ROC-AUC, PR-AUC,
  log loss; binary + multiclass; None for undefined metrics)
- `backend/requirements.txt` (+ torch)
- Tests: models 32, evaluation 11; full suite 113 passing

## Next Files (backend)

- `federated/` — Flower (FedAvg) server + client wrapping
  `TabularClassifier` / `ImageClassifier` / `FusionClassifier`; flwr not
  yet installed in the CrewAI venv
- consume `CSVPipeline` output directly (accepts DataFrame today)
- federated + privacy metrics (proposal §12) deferred

## Design Notes

- Models consume preprocessing outputs; no preprocessing inside models.
- `ImageClassifier` accepts `(N, H, W, C)` (from `ImagePipeline`) or
  `(N, C, H, W)`; auto-detects layout from the channel axis.
- `FusionClassifier.fit/predict` accept a `FusionResult` or a raw 2D
  fused matrix.
- `evaluate_classifier(model, X, y_true)` scores any fitted `BaseModel`
  via `predict` / `predict_proba`; AUC metrics omitted when the target
  or scores cannot support them (single observed class, no proba).
- Reproducibility: `MODEL_RANDOM_SEED` in `models/config.py`; the image
  model seeds RNG and its dataloader shuffle generator.
- Testing requires sklearn + torch: use the CrewAI venv
  (`backend/CrewAI/.venv-opencode`), which has both.
- Existing `CrewAI/app/models/*` are old demos; do not mix with
  `backend/models/`.

## Status

Milestone 2 models + evaluation complete. Federation is the next step;
flwr must be added to the CrewAI venv first.