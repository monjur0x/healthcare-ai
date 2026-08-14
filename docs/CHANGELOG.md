# Changelog

## [Unreleased]

### Added

- `backend/preprocessing/csv/` module: `validator.py`, `cleaner.py`,
  `imputer.py`, `encoder.py`, `feature_engineering.py`, `scaler.py`,
  `transformer.py`, `pipeline.py`, and package `__init__.py`.
- Unit tests for the CSV preprocessing module (`tests/test_csv.py`,
  27 tests passing).
- `backend/pyproject.toml` with shared Black / Ruff / isort settings.
- `backend/preprocessing/image/` module: `validator.py`, `loader.py`,
  `augmentation.py`, `normalization.py`, `pipeline.py`,
  `preprocessing.py`, and package `__init__.py`.
- Image settings in `config.py`: resize, normalization mode (minmax /
  zero_mean / standard), mean/std defaults, augmentation flags.
- `ImageNormalizationError` and `ImageAugmentationError` exceptions.
- Unit tests for the image preprocessing module (`tests/test_image.py`,
  31 tests passing).
- `backend/preprocessing/multimodal/` module: `metadata.py` (sample and
  image metadata schemas), `fusion.py` (concatenate fusion with summary
  / flatten image reduction), and package `__init__.py`.
- Multimodal settings (`FUSION_MODE`, `FUSION_IMAGE_REDUCTION`) in
  `config.py` and `FusionError` exception.
- Unit tests for the multimodal preprocessing module
  (`tests/test_multimodal.py`, 12 tests passing).
- `backend/models/` module: `base.py` (abstract model interface),
  `exceptions.py`, and `csv/tabular.py` (`TabularClassifier` wrapping
  sklearn gradient boosting / logistic / mlp with joblib persistence).
- `backend/requirements.txt` covering backend dependencies.
- Unit tests for the models module (`models/tests/test_tabular.py`,
  10 tests passing).
- `backend/models/config.py` with `ModelSettings` (seed, image training
  hyperparameters, device; env prefix `MODEL_`).
- `backend/models/image/` module: `cnn.py` (`ImageClassifier`, a torch
  CNN with adaptive pooling) and package `__init__.py`. Accepts
  channels-last `(N, H, W, C)` or channels-first batches, trains
  deterministically with a seeded dataloader, persists via
  `torch.save` / `load`.
- `backend/models/multimodal/` module: `fusion_model.py`
  (`FusionClassifier` = MLP over `FusionResult.fused`, composing
  `TabularClassifier`) and package `__init__.py`.
- Unit tests for the new model modules (`models/tests/test_cnn.py`,
  12 passing; `models/tests/test_fusion_model.py`, 10 passing).
- Added `torch` to `backend/requirements.txt`.
- `backend/evaluation/` module: `metrics.py`
  (`ClassificationMetrics` dataclass, `classification_metrics` for
  accuracy / macro precision-recall-F1 / MCC / ROC-AUC / PR-AUC / log
  loss, and `evaluate_classifier` for uniform scoring of any fitted
  `BaseModel`), package `__init__.py`, and unit tests
  (`tests/test_metrics.py`, 11 passing).
- Weight exchange on models: `get_parameters` / `set_parameters`
  (tabular logistic/MLP, fusion, and CNN via torch state dict) and
  `partial_fit` (incremental MLP training) for federated learning.
- `backend/federated/` module: `parameters.py` (`average_weights`
  FedAvg aggregation), `client.py` (`FederatedClient`, a flwr 1.33
  `NumPyClient` with warm start and per-round local training), package
  `__init__.py`, and unit tests (`tests/test_parameters.py`,
  `tests/test_client.py`, 18 passing).
- Added `flwr` to `backend/requirements.txt`.

### Changed

- Replaced sklearn-based scaling with a dependency-light NumPy
  implementation in `scaler.py`.
- Standardized tooling on `backend/pyproject.toml`.
- Aligned `[tool.isort]` `lines_between_types = 1` with the Ruff isort
  rule so both tools agree on import formatting.
- `TabularClassifier` and the new image/fusion models now read the
  random seed from `models.config` (`MODEL_RANDOM_SEED` via
  `models/config.py`) instead of `preprocessing.config`.
- `TabularClassifier.get_parameters` now returns interleaved
  `coefs_`/`intercepts_` (alternating W/b) so round-tripping through
  `set_parameters` is self-consistent for MLP and logistic.

### Fixed

- Normalized column handling so all stages operate on lowercase
  snake_case column names.
- Graceful skip of unavailable feature-engineering features instead of
  hard failure (configurable via `strict=True`).
- Import sorting in `logger.py` to satisfy both Ruff and isort.