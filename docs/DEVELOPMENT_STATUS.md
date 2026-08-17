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

Milestones for `federated/`, `rag/`, `evaluation/`, CrewAI orchestration,
the FastAPI `api/`, `n8n/`, the frontend Streamlit dashboard, and the
functional end-to-end system (Milestone 8) are complete.

---

## Milestone 8 — Functional end-to-end system

### API training path

- [x] `POST /api/v1/train` — train a tabular model through the API and
      serve it immediately (no manual CLI step):
  - [x] Presets (`diabetes` / `heart` / `kidney` / `sepsis`) or explicit
        `dataset` + `target`
  - [x] Central fit (default) and federated FedAvg path
        (`federated: true` + `clients` / `rounds`)
  - [x] Hold-out metrics (accuracy / ROC-AUC / macro F1) returned with
        the artifact path; federated round metrics when federated
  - [x] `APISettings` gains `ARTIFACTS_DIR` / `DATASET_DIR`
  - [x] Tests: +14 (route validation, central + federated training,
        error mapping) — API suite now 33 passing

### n8n — single end-to-end workflow

- [x] `n8n/healthcare-endtoend.json` — one workflow automates the full
      lifecycle: webhook → (optional) `train` → `analyze` → write report
      JSON to disk → structured success/error response
- [x] Removed `clinical-pipeline-modality.json` (superseded by the
      end-to-end workflow); `clinical-analysis.json` kept as a minimal
      reference example
- [x] `n8n/README.md` updated with the new workflow, payloads, and smoke
      test

### Runner + documentation

- [x] `scripts/run_system.sh` — one-command `start` / `status` / `stop`
      (trains default model, starts backend + dashboard, starts n8n via
      Docker; `N8N_ENABLED=0` to skip n8n)
- [x] Root `README.md` — full step-by-step run guide for CPU-only
      machines
- [ ] Full OAuth / per-user auth (static `API_TOKEN` only; backlog)
- [ ] File-upload endpoint for CSV / image inference (backlog)

### Verification

- [x] Backend suite 255 passing; frontend suite 9 passing
- [x] Live end-to-end on diabetes: `/train` central (accuracy ~0.81),
      `/train` federated (2 clients / 2 rounds, full `federated_metrics`),
      `/predict`, `/retrieve`, `/analyze` (prediction + risk + 3 evidence
      items), dashboard running
- [x] n8n workflow JSON + embedded Code-node JS validated

---

## Milestone 8.1 — Image analysis + friendly dashboard input

### Image path (backend)

- [x] `ImageClassifier` now accepts string class labels (mapped by
      `np.unique` order instead of `int(label)`) so the brain-tumor
      classes (`glioma` / `meningioma` / `notumor` / `pituitary`) work
      out of the box
- [x] `run_image_prediction` — orchestrator service that predicts from a
      preprocessed `(H, W, C)` image array and builds a `PredictionResult`
- [x] `ClinicalCrew` gains an `image_model` / `image` path (prediction →
      risk → evidence → report) alongside the tabular path
- [x] API: `GET /api/v1/model` (available models, type, classes,
      feature names) and `POST /api/v1/analyze/image` (base64 image →
      image model → report)
- [x] `AnalysisService` gains `image_model` (loaded from
      `API_IMAGE_MODEL_PATH`) + `model_info()` + `analyze_image()`
- [x] `APISettings.IMAGE_MODEL_PATH` configuration
- [x] `scripts/train_image_model.py` — trains the brain-tumor CNN
      (folder-per-class dataset, hold-out metrics, artifact written to
      `backend/artifacts/brain/global_model.pt`)
- [x] `scripts/run_system.sh` wires `API_IMAGE_MODEL_PATH` from the
      trained artifact

### Dashboard UX

- [x] Clinical Analysis: per-feature numeric inputs when the backend
      reports `feature_names` (no more JSON typing); raw JSON kept as a
      fallback when no model is configured
- [x] Clinical Analysis: **Image (MRI upload)** mode — `file_uploader`
      with live preview → `analyze_image` → shared report layout
- [x] Prediction tab: same friendly per-feature form
- [x] Info tab: shows model metadata + new endpoints
- [x] Client: `model_info()` + `analyze_image()` (base64)

### Verification

- [x] Backend suite 271 passing (+16: image prediction, crew image path,
      model info, analyze image, CNN string labels)
- [x] Frontend suite 13 passing (+4: client model info, analyze image
      x2, image-upload smoke)
- [x] Live: trained brain-tumor CNN (4 classes, hold-out accuracy ~0.71,
      ROC-AUC ~0.92); real glioma scan → predicted `glioma` at ~78%
      confidence with risk + 3 evidence items; tabular analyze unchanged
- [x] Lint clean (black / ruff / isort) across api, orchestrator, models,
      frontend, scripts

---

## Milestone 9 — Privacy-preserving federated learning (paper §8)

### Privacy layer (`backend/federated/privacy.py`)

- [x] `PrivacyConfig` — DP hyperparameters (noise multiplier, grad-norm
      clipping, delta, local epochs, batch, lr)
- [x] `anonymize_frame` — removes PII-like columns (`PII_PATTERNS`):
      name / patient / dob / ssn / phone / email / address / zip / ...
- [x] `pseudonymize` — deterministic truncated SHA-256 pseudonyms
- [x] `train_with_differential_privacy` — Opacus DP-SGD over a torch
      module + epsilon audit via `PrivacyEngine.get_epsilon`
- [x] `SecureAggregator` — pairwise one-time-pad SecAgg adapted to the
      repo's `list[np.ndarray]` param format; masks cancel to the exact
      mean (equal weights required)
- [x] `membership_inference_auroc` — confidence-based MIA simulator
      (members vs hold-out), sorting fixed so AUROC is orientation-correct
- [x] `data_leakage_rate` + `privacy_metrics_summary` — epsilon,
      budget-used %, MIA-AUROC, attack-resistance score (clamped [0, 1]),
      leakage rate, mechanism string
- [x] Unit tests (`federated/tests/test_privacy.py`, 12 passing)

### Federated wiring

- [x] `models/csv/TorchMLPClassifier` — torch MLP with
      `get_parameters` / `set_parameters` / `partial_fit` on the same
      contract as `TabularClassifier` (required because Opacus needs a
      `torch.nn.Module`); saved/loaded via the existing joblib payload
      format (`kind="torch_mlp"`)
- [x] `FederatedClient` accepts a `PrivacyConfig`; local training uses
      DP-SGD and reports the per-round epsilon in fit metrics
- [x] `FedAvgServer` accepts `secure_aggregation=True` (uses
      `SecureAggregator` instead of `average_weights`) and surfaces
      `differential_privacy` / `secure_aggregation` / `epsilon`
      (worst-case per-client epsilon) on `server.metrics`
- [x] `FederatedMetrics` extended with `secure_aggregation`,
      `differential_privacy`, `epsilon`

### API

- [x] `TrainRequest` gains `differential_privacy`, `noise_multiplier`,
      `max_grad_norm`, `privacy_delta`, `secure_aggregation`
- [x] `AnalysisService.train` / `_train_federated` build a torch MLP per
      client when DP is on, collect epsilons, run the MIA audit
      (members = client shards, non-members = hold-out), and append the
      `federated_metrics.privacy` block
- [x] `POST /api/v1/train` wired end-to-end; live-verified on the
      diabetes preset (ε ≈ 2.14, 53.5% of the ε=4 budget,
      MIA-AUROC ≈ 0.50, attack-resistance ≈ 1.0, leakage 0.0, DP +
      SecAgg both reported)

### Verification

- [x] Backend suite 289 passing (+18: privacy module, DP client, SecAgg
      server, DP+SweCgg federated train, torch MLP roundtrip)
- [x] Frontend suite 13 passing (unchanged)
- [x] Lint clean (black / ruff / isort)

---

## Milestone 10 — Evaluation-gap closure (paper §12)

### Differential privacy dependency

- [x] `opacus>=1.5.0` declared in `backend/requirements.txt` under a
      `# Differential privacy` section (was installed but undeclared)
- [x] DP test path exercised against the real Opacus API (green)

### Persistent vector store (`backend/rag`)

- [x] `store_chroma.py` — `ChromaVectorStore` (persistent ChromaDB,
      cosine-only, `EmptyCorpusError` on empty search) with the same
      `add` / `search` / `__len__` interface as the in-memory `VectorStore`
- [x] `store.py::build_vector_store()` factory selects `memory` (default)
      / `chroma` via `RAG_VECTOR_STORE`; `CHROMA_PERSIST_DIR` /
      `CHROMA_COLLECTION` configure persistence
- [x] Tests (`rag/tests/test_vector_store_chroma.py`) — temp-dir
      isolation, persist-across-reopen, cosine-only rejection

### Dense embedding (`backend/rag`)

- [x] `embedder.py::SentenceTransformerEmbedder` — dense embeddings
      (default `BAAI/bge-small-en-v1.5`), lazy model load, BGE query
      instruction, `EmbeddingError` when the dependency is missing
- [x] `build_embedder("sentence-transformer")` opt-in via
      `RAG_EMBEDDING_MODEL`; TF-IDF stays the default
- [x] Tests (`rag/tests/test_embedder_sentence_transformer.py`) —
      graceful skip when the model is unavailable offline

### RAGAS-style generation metrics (`backend/rag`)

- [x] `context_precision`, `context_recall`, `faithfulness`,
      `answer_relevancy` (LLM-free, embedder-agnostic)
- [x] `RAGQualityMetrics` dataclass + `rag_quality_metrics()` aggregator
      with `to_dict()`
- [x] Tests (`rag/tests/test_rag_metrics.py`, 13 passing)

### Agent metrics (`backend/CrewAI/orchestrator`)

- [x] `metrics.py` — `task_completion_rate`, `decision_consistency`,
      `agent_collaboration_score`; `compute_agent_metrics()` →
      `AgentMetrics` with `to_dict()`
- [x] `ClinicalReport` optional `agent_metrics` block wired through
      `assemble_clinical_report`
- [x] Tests (`tests/test_agent_metrics.py`, 14 passing)

### Transport-security decision (docs only)

- [x] ADR-014 — encrypted communication is a deployment-layer concern
      (TLS/mTLS at a reverse proxy); no application code change
- [x] README "Privacy & Security" section (data protection / access
      control / transport security + secrets policy)

### Agentic report parsing (Milestone 10 follow-up)

- [x] `RiskResult.monitoring_schedule` tolerant `field_validator` —
      coerces LLM string entries into `{test, frequency}` dicts so
      agentic output lands instead of the silent deterministic fallback
- [x] `REPORT_SCHEMA_INSTRUCTIONS` shows the `{test, frequency}` shape
      with a bare-string warning
- [x] Live-verified on `gemini-3.6-flash`: "LLM analysis complete",
      agentic summary / recommendations / 7 schedule entries in the
      report; `gemini-3.7-flash` still transient 503 (open question)
- [x] Tests: backend 331 (+5), lint clean

### Verification

- [x] Backend suite 326 passing (+37), frontend 13 (unchanged)
- [x] Lint clean (black / ruff / isort) — run from `backend/` so
      `backend/pyproject.toml` covers the frontend too
- [x] Committed as: `fix(federated)` opacus dep, `feat(rag)` ChromaDB
      store, `feat(rag)` sentence-transformer embedder, `feat(rag)` RAGAS
      metrics, `feat(crew)` agent metrics, `docs(security)` ADR-014,
      `style(rag)` lint pass

---

## Milestone 11 — Doctor-facing CDS dashboard (Milestone 7 rework)

Scope: turn the research-facing Streamlit dashboard into a doctor-friendly
Clinical Decision Support interface aligned with the research workflow
(Patient Data → Federated Prediction → Disease Prediction Agent → RAG →
Treatment Agent → Explainability → n8n → Doctor Dashboard).

### Shared view-layer module

- [x] `frontend/dashboard/clinical.py` — pure presentation helpers (no
      Streamlit, unit-testable):
  - `group_features` — research-defined groups (Vital Signs / Clinical
    Measurements / Medical History / Additional Model Features)
  - `feature_label`, `feature_bounds`, `is_flag_feature` — display
    labels, safe numeric ranges, binary-flag detection
  - `build_analyze_payload`, `analysis_stages`,
    `explanation_sections`, `output_availability` — request payload,
    post-hoc pipeline stages (only real report fields are marked done),
    and the derived explainable decision report
- [x] Unit tests (`tests/test_clinical.py`, 14 passing)

### Client (`frontend/dashboard/client.py`)

- [x] `analyze_via_n8n()` — POST the analysis payload to the n8n
      end-to-end webhook (`/webhook/healthcare-endtoend`) and read the
      full `ClinicalReport` back from `body["report"]`
- [x] `n8n_health()` — probe `{n8n_base}/healthz` for the route
      decision; shared `_analyze_payload()` used by direct and n8n paths
- [x] Tests (+6: webhook success / missing-report / workflow error /
      HTTP error, healthz ok / down)

### Dashboard UI (`frontend/streamlit_app.py`)

- [x] Five tabs: **Overview**, **Clinical Assessment**, **Imaging**,
      **Results**, **System Status**
- [x] Model-driven assessment form: only features the backend reports
      (`/api/v1/model`) are shown, grouped, with flags as checkboxes and
      `sex`/`gender` as 0/1 selectors; one **Analyze Patient** action
- [x] Analysis routing: `Automatic` (n8n when reachable, else direct) /
      `Via n8n workflow` / `Direct to FastAPI`, chosen in the sidebar
      "Advanced" expander (`N8N_ENABLED=0` is the dev-only direct route)
- [x] Imaging page: upload → preview → analyze when an image model is
      configured; honest "not currently available" state otherwise
- [x] Results renderer: the six research outputs — Disease Risk Score
      (score / level badge / confidence / contributing factors /
      monitoring), Mortality Risk and Readmission Risk (explicitly "not
      estimated — future work"), Treatment Recommendation (only from the
      report; honest message when the Treatment Agent produced none),
      Clinical Evidence (source label + text, no vector ids / scores),
      Explainable Decision Report (derived from actual prediction /
      risk outputs, no chain-of-thought)
- [x] Analysis pipeline stages derived post-hoc from the report (no fake
      progress); report JSON download
- [x] System Status: live probes of FastAPI, ML model, RAG, CrewAI, and
      n8n + the current effective route
- [x] No patient persistence — documented as future work (each
      assessment is entered fresh)

### n8n integration

- [x] `healthcare-endtoend.json` Code node returns the full `report` in
      the webhook response so the dashboard consumes the real report
      through the n8n path (workflow JSON validated)
- [x] Blood Pressure input in the Clinical Assessment form accepts
      `SYS/DIA` (e.g. `120/90`); the model's `bloodpressure` feature is
      the diastolic reading, so `120/90` → `90` (`parse_blood_pressure`,
      +3 tests, invalid entries error with a `80` fallback)

### Live n8n end-to-end verification (follow-up)

- [x] Verified live against a real n8n 2.34.6 instance + the real
      backend (previously only hermetic tests + JSON validation):
- [x] Fixed `healthcare-endtoend.json` field references: the n8n webhook
      node nests the payload under `body`, so expressions now read
      `$json.body.*` / `item.json.body.*` (previously `train`, `patient`,
      `features` were silently undefined → train never ran, patient
      "Unknown")
- [x] Removed the `Write: Report to Disk` node from the response
      critical path — n8n 2.34 readWriteFile is sandboxed to
      `~/.n8n-files`, cannot create parent dirs, and its failure produced
      an empty HTTP 200; the full report is returned directly by the
      Respond node
- [x] Respond nodes use `firstIncomingItem` (single JSON object) so the
      dashboard's `analyze_via_n8n()` contract holds
- [x] `clinical-analysis.json` Respond nodes aligned
- [x] Removed the `Merge: Merge Errors` node — with a single input it
      emitted nothing (empty webhook body); each error formatter
      (`Code: Format Train Error`, `Code: Format Analyze Error`) and the
      validation IF now respond directly to `Respond to Webhook (Error)`
- [x] Live results: `analyze_via_n8n()` returns the report with the real
      patient; `train: true` through the webhook trains a diabetes
      logistic model (accuracy 0.66) and returns prediction + risk +
      evidence in one response; missing-feature inputs return a readable
      `status: error` payload
- [x] Original `healthcare-n8n` instance (:5678) is live with the fixed
      workflow: `Healthcare API Token` credential (id
      `6bjqNVT4MoPaTZ6L`), workflow id
      `e2f0a94c-90ee-4f9f-9b39-4d6bfd71b4e2` deployed (URLs →
      `172.17.0.1:8000`) + activated; analyze-only, train+analyze
      (accuracy 0.683), and error paths verified over the webhook and
      via `analyze_via_n8n()`

### Verification

- [x] Frontend suite **35 passing** (+22: clinical helpers 14, client
      n8n 6, app smoke rewritten 7), backend suite 326 passing
      (unchanged, no backend edits)
- [x] Lint clean (black / ruff / isort) — note `frontend/pyproject.toml`
      now mirrors the backend tooling config so the frontend lints
      standalone (ruff config resolves from cwd; black/isort from file
      location)
- [x] Live against a running backend: `/api/v1/model` returns the
      brain-tumor CNN (image only), `analyze` with no tabular model
      returns evidence-without-prediction gracefully, and a real glioma
      scan via `analyze_image` returns prediction (meningioma @ 69%) +
      risk + evidence through the dashboard client
- [x] Live n8n path verified (follow-up): the end-to-end webhook runs
      against a real n8n instance + backend — see "Live n8n end-to-end
      verification" above (was "not live-tested" previously)

---

## Milestone 12 — Baseline comparison study (paper §13)

Scope: run the five proposal configurations together as a structured
comparison on the shipped datasets, reusing the existing metric modules
(no reimplementation).

- [x] `backend/scripts/baseline_study.py` — standalone runner:
  1. Centralized ML (`AnalysisService.train(federated=False)`)
  2. Federated-only (`train(federated=True, clients=3, rounds=5)`) +
     `FederatedMetrics` (comm. cost via `parameter_set_bytes`, convergence
     via the round-accuracy history in `federated_metrics`)
  3. Federated + RAG — same model, `rag_quality_metrics` over 5 literal
     clinical queries per dataset (reference answers grounded in literal
     per-dataset corpora)
  4. Federated + Multi-Agent — deterministic LLM-free crew
     (`ClinicalCrew`/`AnalysisService.analyze`) over 5 sampled test rows +
     `compute_agent_metrics`
  5. Proposed (full) — agents + RAG evidence wired into the same crew;
     n8n documented as the qualitative orchestration layer (no fabricated
     metric)
- [x] Same held-out split for all baselines (`test_size=0.25`, `seed=42`),
      classification block reused for baselines 2–5 (RAG/agents do not
      retrain); `n/a` for metrics a baseline does not produce
- [x] `docs/BASELINE_STUDY_RESULTS.md` — real numbers (not placeholders) +
      hand-written Findings answering RQ1–RQ4 with pilot-scale caveats;
      the Findings section survives script re-runs
- [x] `prepare_tabular_data` strips whitespace from string labels
      (`'ckd\t'` == `'ckd'`), fixing the kidney preset's phantom third
      class that broke the federated partition (matches
      `examples/fedavg_demo.py`)
- [x] Tests (`backend/scripts/tests/test_baseline_study.py`, 4 passing) —
      hermetic on a synthetic CSV, no `DATASET_DIR` required

### Verification

- [x] Backend suite **335 passing** (+4 study tests; `api`, `scripts`)
- [x] Lint clean (black / ruff / isort) from `backend/`
- [x] Real-data run: `DATASET_DIR=/home/monjur0x0/dataset` → all four
      presets (diabetes / heart / kidney / sepsis) with full tables
- [x] Headline: federated Δ accuracy +0.027 / +0.026 / +0.020 / 0.000 vs
      centralized; RAG context recall 1.000, precision 0.300–0.350; agent
      completion 0.6→0.8 with RAG evidence

---

## Milestone 13 — Doctor-friendly Clinical Assessment page

Scope: make the dashboard's Clinical Assessment page model/preset-driven
and doctor-friendly while keeping it a thin presentation layer (ADR-010):
human labels, verified units, patient context separated from model
features, validation, CSV upload, and honest pre-run summaries. No ML
logic moves into the frontend; the FastAPI + n8n + CrewAI architecture is
unchanged.

### Backend (presets + CSV analysis)

- [x] `GET /api/v1/presets` — `PresetInfo` (name, dataset, target,
      available, feature_names, classes) per shipped preset, derived from
      the trained artifact (`artifacts_dir/<preset>/global_model.joblib`,
      `TabularClassifier.load`); a preset with no artifact is reported
      `available: false`
- [x] `POST /api/v1/analyze/csv` — `AnalyzeCSVRequest` (base64-decoded
      `csv: bytes`, optional patient/markers/recommendations) →
      `CSVPipeline().run(csv)` → first-row analysis through the existing
      `analyze()` path; missing columns → 422, no tabular model → 503
- [x] `ModelInfo.preset` — the served model's training preset, recorded
      by `AnalysisService.train()` and exposed via `/api/v1/model`

### Dashboard assessment tab (`frontend/streamlit_app.py`)

- [x] **Assessment Type** selector — served preset when a single model is
      configured, otherwise the preset list from `/api/v1/presets`;
      switching presets swaps the feature schema dynamically
- [x] Train-on-demand: when the selected preset differs from the served
      model, the dashboard trains it first (direct route →
      `client.train(preset)`; n8n route → `preset` + `train` in the
      webhook payload, already supported by `healthcare-endtoend.json`)
- [x] **Patient Context** (name / id / age) separated from model features
      and never sent to the model (the `age` model feature is filled from
      the context age in one place)
- [x] Model-driven "Clinical measurements": only backend-reported features,
      grouped, human labels (no raw column names), verified units in
      labels, `%d` formatting for integer features, `sex`/`gender`
      selectors and flag checkboxes
- [x] Blood-pressure widget returns `float | None` — invalid input blocks
      submission with an error instead of silently substituting `80`
- [x] Pre-run **Assessment summary** (patient / assessment / entered
      features / notes), validation before submit, and a clinical
      disclaimer caption on form + results
- [x] Input method toggle **Manual Entry / CSV Upload** — CSV uploads
      route directly to `/api/v1/analyze/csv` (the n8n workflow carries
      structured feature input only); caption states this honestly

### Presentation helpers (`frontend/dashboard/clinical.py`)

- [x] `DISPLAY_LABELS` — human labels for all four presets (Pima / UCI
      heart / UCI CKD / sepsis), raw-name suffixes removed
- [x] `FEATURE_UNITS` (glucose mg/dL, bloodpressure mmHg, bmi kg/m², ...),
      `INTEGER_FEATURES`, `feature_unit()`, `is_integer_feature()`
- [x] `validate_feature_values()` — bounds + required checks (age bounds
      from the Patient Context), used to block invalid submission
- [x] `assessment_summary()` — derived pre-run summary rows

### Client (`frontend/dashboard/client.py`)

- [x] `presets()` — fetch `/api/v1/presets`
- [x] `train(preset, model="mlp")` — `POST /api/v1/train`
- [x] `analyze_csv(patient, csv, ...)` — base64-encoded `POST
      /api/v1/analyze/csv`
- [x] `analyze_via_n8n()` / `_analyze_payload()` accept `preset` / `train`
      kwargs forwarded to the n8n webhook only when a retrain is requested

### Verification

- [x] Frontend suite **54 passing** (+14: client presets/train/analyze_csv
      ×5, clinical labels/units/integers/validation/summary ×6, smoke:
      selector adapts form, train-on-demand direct route, CSV upload ×3)
- [x] Backend suite **340 passing** (+5: presets schema, analyze_csv
      success / missing-CSV 422 / invalid-base64 422 / service error 503)
- [x] Lint clean (black / ruff) from `frontend/` and `backend/`
- [x] Smoke tests cover preset switching (dynamic fields), train-on-demand
      (patient metadata never sent as features), and CSV upload
      (bytes + patient forwarded, direct-route caption)

---

## Milestone 6 — n8n orchestration (`n8n/`)

### Principles

- Orchestration only: n8n triggers workflows and calls the FastAPI
  backend; AI reasoning stays in the CrewAI crew, prediction / retrieval
  stay in `models/` / `rag/` (`AGENTS.md`).

### Workflows

- [x] `clinical-analysis.json` — webhook (`/webhook/healthcare-analyze`)
      → `POST /api/v1/analyze` → validate + summarize → structured
      `status: success|error` response (error branch merges HTTP and
      validation failures)
- [x] `clinical-pipeline-modality.json` — webhook
      (`/webhook/healthcare-pipeline`) → normalize input → switch on
      `modality` (image / csv) → `/api/v1/analyze` with matching
      `input_type` → merged success summary or merged error payload
- [x] Optional bearer-token auth via an `httpHeaderAuth` credential
      (placeholder reference; works while `API_TOKEN` is unset)
- [x] `README.md` — import, configuration, example payloads, local smoke
      test

### Housekeeping

- [x] Removed stale `n8n/workflow.json` / `workflow2.json` (targeted the
      removed old demo endpoints)

---

## Milestone 7 — Streamlit dashboard (`frontend/`)

> **Superseded in part by Milestone 11.** The original research-facing
> tabs (Clinical Analysis / Prediction / Evidence Retrieval / Info) were
> replaced by the doctor-facing Clinical Decision Support layout
> (Overview / Clinical Assessment / Imaging / Results / System Status).
> The thin view-layer principle (ADR-010) and the client tests remain.

### Principles

- Thin view layer only: the dashboard talks to the FastAPI backend and
  renders; all reasoning happens server-side (`backend/api` -> CrewAI
  crew). ADR-010.

### Client (`frontend/dashboard`)

- [x] `client.py` — `HealthcareAPIClient` (httpx): `health`, `predict`,
      `retrieve`, `analyze`; optional bearer token; `HealthcareAPIError`
      with status + code from the backend error detail
- [x] Unit tests (`tests/test_client.py`) — passing via
      `httpx.MockTransport` (hermetic, no network)

### UI (`frontend/streamlit_app.py`)

- [x] Sidebar — backend URL + optional API token + live health indicator
- [x] Tabs: Clinical Analysis (form -> full report with prediction, risk,
      evidence, recommendations, JSON download), Prediction (bar chart
      of probabilities), Evidence Retrieval (score bars), Info (health +
      endpoint reference) — **reworked in Milestone 11**
- [x] AppTest smoke tests (`tests/test_app_smoke.py`) — passing
      (boot + mocked analyze submission renders the report)
- [x] `frontend/requirements.txt` — `streamlit`, `httpx`, `pytest`

### Verification

- [x] Frontend suite: 9 passing; backend suite: 241 passing
- [x] Live end-to-end: uvicorn backend + real client (`/health`,
      `/api/v1/analyze` with evidence, `/api/v1/retrieve`, 503 on
      predict without a configured model)
- [ ] Next.js dashboard (architecture doc mentions it; the Streamlit
      dashboard currently fills the frontend role)

---

## Milestone 5 — FastAPI API (`backend/api`)

### Scaffolding

- [x] `config.py` — `APISettings` (env prefix `API_`): server metadata,
      `MODEL_PATH`, `CORPUS_DIR`, optional `API_TOKEN`, CORS origins
- [x] `exceptions.py` — `APIError` + `ServiceUnavailableError` /
      `InvalidInputError` / `AuthenticationError` / `NotFoundError`
- [x] `schemas.py` — request models (`PredictRequest`, `RetrieveRequest`,
      `AnalyzeRequest`, `HealthResponse`) reusing orchestrator
      `PredictionResult` / `EvidenceItem` / `ClinicalReport` responses

### Service layer

- [x] `services.py` — `AnalysisService` facade: lazy model load,
      RAG corpus ingest (directory or built-in corpus), deterministic
      crew analysis; domain exceptions translated to typed `APIError`s
- [x] `load_predictive_model` / `build_rag_pipeline` helpers

### Routes (validation + delegation only)

- [x] `routes.py` — `/api/v1/predict`, `/api/v1/retrieve`,
      `/api/v1/analyze`; optional bearer-token auth (router dependency)
- [x] `main.py` — `create_app()` factory (DI via app state, CORS,
      `APIError` → JSON handler); module-level `app` for uvicorn
- [x] Unit tests (`tests/test_api.py`, `tests/test_services.py`) —
      19 passing
- [ ] OAuth / full user authentication (currently an optional static
      bearer token; see backlog)

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
- [x] `partial_fit` — one-epoch incremental training from current
      weights; the image path joins federated rounds (ADR-006)
- [x] Deterministic training (seeded RNG + seeded dataloader shuffle)
- [x] Persistence via `torch.save` / `ImageClassifier.load`
- [x] Unit tests (`models/tests/test_cnn.py`) — 16 passing

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
      `test_server.py`, `test_cnn_federation.py`) — 25 passing
- [x] CNN federates end-to-end via `ImageClassifier.partial_fit`
      (ADR-006); end-to-end tests in `federated/tests/`

### Federated metrics (`federated/metrics.py`)

- [x] `parameter_set_bytes` — bytes for a full weight exchange
- [x] `round_accuracy_deltas` + `convergence_round` — round-to-round
      accuracy change and first converged round (threshold-tunable)
- [x] `FederatedMetrics` dataclass (rounds/clients, bytes exchanged,
      per-round + total time, accuracy deltas, convergence round) with
      `to_dict()` for JSON reports
- [x] `FedAvgServer.run()` records per-round wall-clock duration and
      estimated communication bytes (client upload + broadcast);
      exposed via `RoundResult` fields and the `server.metrics` property
- [x] Demo report (`fedavg_demo.py`) now includes `federated_metrics`
- [x] Unit tests (`tests/test_federated_metrics.py`) + server tests —
      10 passing

### End-to-end demo (`backend/examples`)

- [x] `fedavg_demo.py` — CSV → `CSVPipeline` → `TabularClassifier`
      (MLP) → FedAvg rounds → evaluation report. Presets for the local
      datasets: diabetes / heart / kidney / sepsis. Partitions train
      rows into class-balanced client shards (StratifiedKFold), trains
      the synchronous `FedAvgServer`, compares against a central
      baseline, writes `global_model.joblib` + `report.json`.
- [x] `image_fedavg_demo.py` — image-path FedAvg demo. Discovers
      class-labelled image folders (e.g. the brain-tumor MRI dataset),
      preprocesses with `ImagePipeline`, federates the CNN via
      `ImageClassifier.partial_fit`, reports baseline + federated
      metrics, writes `global_model.pt` + `report.json`.
- [x] Smoke tests (`examples/tests/test_image_fedavg_demo.py`) — 3
      passing (run on synthetic image trees, no external data)

---

## Milestone 3 — RAG (Retrieval-Augmented Generation)

Scope: document ingestion, embedding generation, vector search, and
context retrieval, per `docs/SOFTWARE_ARCHITECTURE.md` §rag/.

### Package scaffolding (`backend/rag`)

- [x] Exceptions (`exceptions.py`) — `RAGError` + `EmptyCorpusError`,
      `EmptyQueryError`, `InvalidDocumentError`, `EmbeddingError`,
      `RetrievalError`
- [x] Configuration (`config.py`) — `RAGSettings` (env prefix `RAG_`):
      chunk size/overlap, embedding model, max features, top-k, metric
- [x] Data structures (`documents.py`) — `Document` / `Chunk` /
      `RetrievalResult`, all frozen dataclasses with `to_dict()`

### Retrieval components

- [x] `chunker.py` — `TextChunker`: deterministic word-based sliding
      window with configurable overlap
- [x] `embedder.py` — `Embedder` ABC + `TfidfEmbedder` (corpus-fitted)
      + `HashingEmbedder` (fit-free fixed-dim) + `build_embedder`;
      transformer embedders swappable behind the interface
- [x] `store.py` — `VectorStore`: in-memory NumPy nearest-neighbour
      search over cosine / dot
- [x] `retriever.py` — `Retriever`: incremental ingest, query → top-k
      chunks, `build_context` (source-labelled prompt block)
- [x] `metrics.py` — `precision_at_k`, `recall_at_k`,
      `mean_reciprocal_rank`, `RetrievalMetrics`
- [x] `pipeline.py` — `RAGPipeline` composing chunker → embedder →
      store → retriever (`ingest_documents`, `ingest_texts`,
      `retrieve`, `build_context`)
- [x] No new dependencies (reuses scikit-learn)
- [x] Unit tests (`rag/tests/`) — 38 passing

### Retrieval demo (`backend/examples`)

- [x] `rag_demo.py` — corpus directory → `RAGPipeline` → queries →
      top-k chunks + context + (optional) quality metrics
- [x] Smoke tests (`examples/tests/test_rag_demo.py`) — 2 passing

---

## Milestone 4 — CrewAI Orchestration

Scope: multi-agent reasoning over the outputs of the preprocessing,
prediction, and retrieval modules, per `docs/SOFTWARE_ARCHITECTURE.md`
§CrewAI/.

### Package scaffolding (`backend/CrewAI/orchestrator`)

- [x] Configuration (`config.py`) — `CrewSettings` (env prefix `CREW_`):
      optional LLM provider/model/key, crew verbosity/memory, RAG top-k,
      risk + marker thresholds
- [x] Exceptions (`exceptions.py`) — `CrewError` + tool / report /
      `LLMNotConfiguredError` subclasses
- [x] Schemas (`schemas.py`) — `PatientInfo`, `PredictionResult`,
      `RiskResult`, `EvidenceItem`, `ClinicalReport` (pydantic)

### Deterministic services (`services.py`)

- [x] `run_prediction` — single-row prediction from a fitted
      `TabularClassifier` (feature-aligned, error-guarded)
- [x] `assess_risk` — risk score/level from confidence + clinical
      marker thresholds, with a deterministic monitoring schedule
- [x] `retrieve_evidence` — wraps `RAGPipeline` into `EvidenceItem`s
- [x] `assemble_clinical_report` — final structured report, consistent
      prediction+risk pairing enforced

### CrewAI layer

- [x] `prompts.py` — role/goal/backstory for the seven agents, task
      descriptions, report schema instructions
- [x] `tools.py` — `PredictionTool`, `RiskAssessmentTool`,
      `RAGRetrievalTool`, `ClinicalReportTool` (crewai `BaseTool`
      wrappers over the services)
- [x] `agents.py` — seven agents (Patient Analysis, Disease Prediction,
      Medical Research, Treatment Planning, Explainability, Risk
      Monitoring, Report Writing)
- [x] `tasks.py` — chained tasks (analysis → prediction → evidence →
      treatment → explanation → risk → report)
- [x] `crew.py` — `ClinicalCrew`: `run_analysis()` offline
      deterministic pipeline (ADR-008), `run_llm()` optional CrewAI
      kickoff with deterministic fallback, `run()` selects by config
- [x] Agents/tasks/crew construct without an LLM key (hermetic)
- [x] Unit tests (`CrewAI/orchestrator/tests/`) — 25 passing

### Orchestration demo (`backend/examples`)

- [x] `clinical_crew_demo.py` — CSV → `CSVPipeline` →
      `TabularClassifier` → `ClinicalCrew.run_analysis()` → clinical
      report (`report.json`); built-in or `--corpus-dir` knowledge base
- [x] Smoke tests (`examples/tests/test_clinical_crew_demo.py`) —
      2 passing

### Cleanup (`backend/CrewAI`)

- [x] Removed the superseded old demo (`app/`, `tests/test_healthcare.py`,
      `Dockerfile`, `docker-compose.yml`, `requirements.txt`,
      `.env.example`, `README.md`) — nothing referenced it and the new
      modules supersede it
- [x] Untracked generated artifacts (`artifacts/*.pt`, `artifacts/*.json`)
      and ignored `artifacts/` via `.gitignore`
- [ ] Differential privacy port from the old demo (recorded in backlog)

---

## Testing

- [x] Preprocessing: 70 tests passing
- [x] Models: 36 tests passing (tabular 10 / CNN 16 / fusion 10)
- [x] Evaluation: 11 tests passing
- [x] Federated: 51 tests passing
- [x] RAG: 61 tests passing (incl. Chroma store, sentence-transformer
      embedder, RAGAS metrics)
- [x] Orchestration (CrewAI): 49 tests passing (incl. agent metrics,
      LLM-style report parsing)
- [x] Examples: 7 tests passing
- [x] API: 51 tests passing (incl. presets + CSV analysis endpoints)
- [x] Scripts (baseline study): 4 tests passing (hermetic on synthetic data)
- [x] Full suite: 340 backend tests passing (`pytest preprocessing/tests
      models/tests evaluation/tests federated/tests rag/tests
      examples/tests CrewAI/orchestrator/tests api/tests
      scripts/tests`)
- [x] Frontend: 54 tests passing (from `frontend/`)
- [ ] Full test command documented in README/AGENTS (see `AGENTS.md` tooling note)