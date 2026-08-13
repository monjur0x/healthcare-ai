# Current Context

## Current Milestone

Milestone 2 — Models (in progress)

## Current Module

backend/models (shared, csv, image, multimodal complete)

## Current Task

Shared interface, tabular, image (CNN), and multimodal (fusion) models
are complete. Next: the Flower federated tie-in wrapping these models.

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
- `backend/requirements.txt` (+ torch)
- Model tests: 32 passing (tabular 10 / CNN 12 / fusion 10);
  full suite 102 passing

## Next Files (backend)

- `federated/` — Flower (FedAvg) server + client wrapping
  `TabularClassifier` / `ImageClassifier` / `FusionClassifier`
- `evaluation/` — ROC-AUC / PR-AUC / MCC hooks on `predict_proba`
- consume `CSVPipeline` output directly (accepts DataFrame today)

## Design Notes

- Models consume preprocessing outputs; no preprocessing inside models.
- `ImageClassifier` accepts `(N, H, W, C)` (from `ImagePipeline`) or
  `(N, C, H, W)`; auto-detects layout from the channel axis.
- `FusionClassifier.fit/predict` accept a `FusionResult` or a raw 2D
  fused matrix.
- Reproducibility: `MODEL_RANDOM_SEED` in `models/config.py`; the image
  model seeds RNG and its dataloader shuffle generator.
- Testing requires sklearn + torch: use the CrewAI venv
  (`backend/CrewAI/.venv-opencode`), which has both.
- Existing `CrewAI/app/models/*` are old demos; do not mix with
  `backend/models/`.

## Status

Milestone 2 models (shared/csv/image/multimodal) complete. Federation
and evaluation are the remaining scoped items.