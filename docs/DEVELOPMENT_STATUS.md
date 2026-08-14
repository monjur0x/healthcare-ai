# Development Status

## Legend

- [x] Implemented and tested.
- [ ] Not started.

---

## Milestone 1 — Preprocessing

### Package scaffolding

- [x] Package structure (`backend/preprocessing/__init__.py`)
- [x] Global configuration (`config.py`)
- [x] Centralized logging (`logger.py`)
- [x] Custom exceptions (`exceptions.py`)

### CSV preprocessing (`backend/preprocessing/csv`)

- [x] Validator (`validator.py`)
- [x] Cleaner / column normalization (`cleaner.py`)
- [x] Missing value imputation (`imputer.py`)
- [x] Categorical encoding (`encoder.py`)
- [x] Feature engineering (`feature_engineering.py`)
- [x] Scaling (`scaler.py`)
- [x] Transformer / pipeline orchestration (`transformer.py`)
- [x] High-level entry point (`pipeline.py`)
- [x] Unit tests (`tests/test_csv.py`) — 27 passing

### Image preprocessing (`backend/preprocessing/image`)

- [x] Validator (`validator.py`)
- [x] Loader (`loader.py`) — PNG/JPG via Pillow, DICOM via optional `pydicom`
- [x] Augmentation (`augmentation.py`) — deterministic, seeded
- [x] Normalization (`normalization.py`) — minmax / zero_mean / standard
- [x] Pipeline (`pipeline.py`) — load → validate → resize → augment → normalize
- [x] Convenience API (`preprocessing.py`) — single image, batch, directory
- [x] Unit tests (`tests/test_image.py`) — 31 passing

### Multimodal preprocessing (`backend/preprocessing/multimodal`)

- [x] Fusion (`fusion.py`) — concatenate + summary/flatten image reduction
- [x] Metadata (`metadata.py`) — `SampleMetadata` / `ImageInfo` schemas
- [x] Unit tests (`tests/test_multimodal.py`) — 12 passing

---

## Tooling

- [x] `backend/pyproject.toml` with shared Black / Ruff / isort settings
- [ ] Lint/format/test commands documented before every session (see `AGENTS.md`)

---

## Not yet planned

Milestones for `federated/`, `rag/`, `evaluation/`, `api/`
are defined at the repository level but not yet scoped in the backlog.

---

## Milestone 2 — Models

### Shared (`backend/models`)

- [x] Model interface (`base.py`) — fit / predict / predict_proba / save / load
- [x] Model exceptions (`exceptions.py`)
- [x] Unit tests (`models/tests/test_tabular.py`)

### CSV / tabular (`backend/models/csv`)

- [x] `TabularClassifier` (`tabular.py`) — gradient boosting / logistic / MLP
- [x] Persistence via joblib
- [x] Unit tests — 10 passing

### Image (`backend/models/image`)

- [x] `ImageClassifier` (`cnn.py`) — torch CNN, trains/infers on
      channels-last `(N, H, W, C)` batches
- [x] Adaptive pooling CNN: conv → batch-norm → pool → MLP head
- [x] Deterministic training (seeded RNG + seeded dataloader shuffle)
- [x] Persistence via `torch.save` / `ImageClassifier.load`
- [x] Unit tests (`models/tests/test_cnn.py`) — 12 passing

### Multimodal (`backend/models/multimodal`)

- [x] `FusionClassifier` (`fusion_model.py`) consuming `FusionResult`
      directly (or raw fused matrix); MLP over fused features
- [x] Composes `TabularClassifier` (DRY), joblib persistence
- [x] Unit tests (`models/tests/test_fusion_model.py`) — 10 passing

### Model configuration (`backend/models/config.py`)

- [x] `ModelSettings` — seed, image epochs / batch size / learning rate /
      device; env prefix `MODEL_`

### Evaluation (`backend/evaluation`)

- [x] `metrics.py` — `ClassificationMetrics` dataclass (accuracy,
      precision/recall/F1 macro, MCC, ROC-AUC, PR-AUC, log loss)
- [x] `classification_metrics(y_true, y_pred, y_score, labels)` — pure
      function, binary + multiclass, graceful None for undefined metrics
- [x] `evaluate_classifier(model, X, y_true)` — uniform scoring of any
      fitted `BaseModel` (tabular / image / fusion)
- [x] Unit tests (`tests/test_metrics.py`) — 11 passing

### Federated (`backend/federated`)

- [x] `parameters.py` — `average_weights` (element-wise FedAvg)
- [x] `client.py` — `FederatedClient` (flwr 1.33 `NumPyClient`): warm
      start, one local `partial_fit` per round, log-loss + accuracy eval
- [x] Weight exchange on models (`get_parameters` / `set_parameters`)
      for tabular logistic/MLP, fusion, and CNN; `partial_fit` for MLP;
      `set_parameters` materializes unfitted estimators via dummy fit
- [x] `server.py` — synchronous `FedAvgServer` (init weights, per-round
      client fit, aggregate, evaluate) + `make_global_evaluator`;
      mirrors flwr `FedAvg` without the Ray process spawn
- [x] Unit tests (`tests/test_parameters.py`, `test_client.py`,
      `test_server.py`) — 23 passing

### End-to-end demo (`backend/examples`)

- [x] `fedavg_demo.py` — CSV → `CSVPipeline` → `TabularClassifier`
      (MLP) → FedAvg rounds → evaluation report. Presets for the local
      datasets: diabetes / heart / kidney / sepsis. Partitions train
      rows into class-balanced client shards (StratifiedKFold), trains
      the synchronous `FedAvgServer`, compares against a central
      baseline, writes `global_model.joblib` + `report.json`.

---

## Testing

- [x] Preprocessing: 70 tests passing
- [x] Models: 32 tests passing (tabular 10 / CNN 12 / fusion 10)
- [x] Evaluation: 11 tests passing
- [x] Federated: 23 tests passing
- [x] Full suite: 136 tests passing (`pytest preprocessing/tests models/tests evaluation/tests federated/tests`)
- [ ] Full test command documented in README/AGENTS (see `AGENTS.md` tooling note)