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

- [x] `ImageClassifier` (torch CNN: conv → batch-norm → pool → MLP head)
- [x] Consumes channels-last `(N, H, W, C)` `ImageResult`-style batches
- [x] Fixed seed + deterministic dataloader for reproducibility
- [x] Persistence via `torch.save` / `load`
- [x] Unit tests
- [ ] Pretrained backbones (EfficientNetV2 / DenseNet / Swin-T) —
      deferred; current CNN is dependency-light and offline-friendly

## Multimodal

- [x] `FusionClassifier` consuming `FusionResult` from preprocessing
- [x] MLP (sklearn) over the fused feature matrix by default
- [x] Unit tests

## Evaluation hooks

- [x] `evaluation/metrics.py` — `ClassificationMetrics` + `classification_metrics`
- [x] `evaluate_classifier(model, X, y_true)` scores any fitted `BaseModel`
- [x] Accuracy, precision/recall/F1 (macro), MCC, ROC-AUC, PR-AUC, log loss
- [x] Unit tests
- [ ] Federated metrics (communication cost / convergence / training time)
      and privacy budget — deferred to federated milestone

## Federated

- [x] `federated/parameters.py` — `average_weights` (FedAvg), NumPy-native
- [x] `federated/client.py` — `FederatedClient` (flwr `NumPyClient`,
      flwr 1.33.0) with warm start, one local partial-fit per round,
      log-loss + accuracy evaluation
- [x] Weight exchange on models: `get_parameters` / `set_parameters`
      (tabular logistic/MLP, fusion, CNN via state dict);
      `partial_fit` (MLP)
- [x] Unit tests (roundtrip, FedAvg round, client evaluate)
- [ ] Full flwr server strategy / simulation driver
- [ ] Federate the CNN end-to-end (needs a torch `partial_fit`)

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