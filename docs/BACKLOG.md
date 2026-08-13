# Milestone 1 — Preprocessing (complete)

## Preprocessing

### Scaffolding

- [x] Package structure

- [x] Config

- [x] Exceptions

- [x] Logger

### CSV

- [x] CSV Validator

- [x] CSV Cleaner

- [x] Missing Value Imputer

- [x] Encoder

- [x] Feature Engineering

- [x] Scaler

- [x] Transformer

- [x] Pipeline

- [x] Unit tests

### Image

- [x] Validator

- [x] Loader

- [x] Augmentation

- [x] Normalization

- [x] Pipeline

### Multimodal

- [x] Fusion

- [x] Metadata

---

# Milestone 2 — Models

Scope derived from `ai-automation-research.md` (§11 Proposed Methodology)
and `workflow.txt` (Phase 3): models consume preprocessing outputs, are
trained locally per hospital, and are aggregated by Flower (FedAvg).

## Shared

- [x] Model interface (`models/base.py`) — fit / predict / predict_proba / save / load
- [x] Model exceptions (`models/exceptions.py`)
- [x] Fixed seeds for reproducibility
- [x] Unit tests

## CSV / tabular

- [x] `TabularClassifier` (sklearn: gradient boosting / logistic / MLP)
- [ ] Consume `CSVPipeline` output directly (accepts preprocessed DataFrame)
- [x] Persistence to `artifacts/` via joblib
- [x] Unit tests

## Image

- [ ] Vision models (EfficientNetV2 / DenseNet / Swin-T) — requires torch
- [ ] Unit tests

## Multimodal

- [ ] Model consuming `FusionResult` from preprocessing
- [ ] Unit tests

## Evaluation hooks

- [ ] Expose predict_proba for ROC-AUC / PR-AUC / MCC (proposal §12)

---

## Backlog

### Preprocessing enhancements

- [ ] Add datetime parsing to feature engineering.
- [ ] Add robust error-reporting structure to `CSVPipeline` (~ `valid_frame` etc.).
- [ ] Expose a CLI or fit/persist for scaler/encoder parameters (reproducibility).
- [ ] Persist image normalization statistics (mean/std) for inference-time
      consistency (currently stateless fallback for `standard` mode).
- [ ] Add DICOM unit tests; requires adding `pydicom` to dependencies.
- [ ] Consider aspect-ratio-preserving (letterbox) resize option in
      `ImageLoader` (currently exact square resize).

### Milestone 3+ (not yet scoped)

- [ ] `federated/`, `rag/`, `evaluation/`, `api/` backlog entries.