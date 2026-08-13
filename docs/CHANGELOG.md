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

### Changed

- Replaced sklearn-based scaling with a dependency-light NumPy
  implementation in `scaler.py`.
- Standardized tooling on `backend/pyproject.toml`.
- Aligned `[tool.isort]` `lines_between_types = 1` with the Ruff isort
  rule so both tools agree on import formatting.

### Fixed

- Normalized column handling so all stages operate on lowercase
  snake_case column names.
- Graceful skip of unavailable feature-engineering features instead of
  hard failure (configurable via `strict=True`).
- Import sorting in `logger.py` to satisfy both Ruff and isort.