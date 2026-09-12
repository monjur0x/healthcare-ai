ADR-001

Use Flower for federated learning.

Reason

Official framework.

Supports simulation.

---

ADR-002

CrewAI instead of LangGraph.

Reason

Better fits explicit multi-agent architecture.

---

ADR-003

Preprocessing is a standalone package.

Reason

Reuse across:

- Training
- Inference
- Flower
- FastAPI

---

ADR-004

Pin federated exchange to Flower 1.x `NumPyClient`.

Reason

- `flwr 1.33.0` installed in the CrewAI venv; `NumPyClient` (no
  `set_parameters`; `get_parameters`/`fit`/`evaluate`) matches the
  1.x API.
- Weights travel as plain NumPy arrays (`coefs_`/`intercepts_` for
  tabular, state dict for torch), so no framework-specific encoders are
  needed.

---

ADR-005

Federated local training uses the MLP estimator only.

Reason

- `LogisticRegression` and gradient boosting lack `partial_fit`, which
  the federated client needs to continue training from the aggregated
  global weights; `MLPClassifier.partial_fit` supports it.
- Tabular/Fusion `get_parameters`/`set_parameters` still work for
  logistic; only the incremental training step requires MLP.

---

ADR-006

`ImageClassifier` supports federated local training via `partial_fit`.

Reason

- The torch CNN can continue training from any weight state, so the
  image path joins federated rounds without changing the
  `FederatedClient` or `FedAvgServer` contracts.
- `partial_fit` mirrors the tabular one-pass contract (single epoch of
  gradient steps reusing the current weights); labels are restricted to
  the classes seen at `fit` time so the exchangeable weight shapes stay
  stable across clients and rounds.
- `get_parameters` returns memory-shared NumPy views of the state dict;
  callers that need a stable snapshot must copy them before the model
  trains further.

---

ADR-007

RAG uses a TF-IDF embedder and an in-memory vector store.

Reason

- Retrieval stays dependency-light (reuses scikit-learn) and fully
  offline-friendly, matching the repository's research constraints
  (reproducibility, no hidden randomness, minimal dependencies).
- An `Embedder` ABC plus `VectorStore` keep the production path
  swappable: dense models (sentence-transformers) and a persistent
  backend (Qdrant) can replace them behind the same interfaces without
  touching `Retriever` / `RAGPipeline`.
- `HashingEmbedder` provides a fit-free fixed-dimension fallback for
  hermetic tests and fully offline setups.
- Chunking is deterministic (word-based sliding window with overlap),
  so re-indexing the same corpus yields identical embeddings.

---

ADR-008

CrewAI orchestration runs a deterministic tool pipeline by default, with
an optional LLM layer.

Reason

- The seven-role crew must work, run end-to-end, and be tested without
  an LLM API key, keeping the framework reproducible and hermetic.
- `ClinicalCrew.run_analysis()` executes the tools directly
  (prediction → risk → evidence retrieval → report assembly) and
  always produces a structured `ClinicalReport`; agents never implement
  ML, they consume pipeline outputs.
- `run_llm()` (enabled only when `CREW_LLM_API_KEY` is set) builds the
  CrewAI agents/tasks/crew from the same prompts and falls back to the
  deterministic report when the crew result cannot be parsed.
- Agents/tasks are constructed without a bound LLM unless the LLM path
  is requested, so building them stays dependency-light (crewai's
  google provider extra is not required).
- Tools are thin wrappers over `orchestrator/services.py`, so the
  deterministic core is testable in isolation.

---

ADR-009

FastAPI routes delegate to a service layer and map errors via typed
`APIError` subclasses.

Reason

- Per `AGENTS.md`, routes never contain business logic. `AnalysisService`
  (in `api/services.py`) owns the orchestration and translates domain
  exceptions (prediction / risk / retrieval / orchestration) into typed
  `APIError`s at the service boundary, so routes and handlers never
  import domain exceptions.
- Each `APIError` carries an HTTP status + machine `code`, and a single
  handler in `create_app()` serializes them to a consistent JSON error
  shape.
- Authentication is an optional static bearer token (`API_TOKEN`),
  enforced by a router dependency and off by default — full OAuth is
  deferred to the backlog.
- The service is injected through `app.state` by `create_app(service=...)`,
  so route tests use a hermetic fake service without touching real
  models or corpora.

---

ADR-010

The frontend is a Streamlit dashboard that acts as a thin client over the
FastAPI backend.

Reason

- The `frontend/` directory was empty; the architecture doc names a
  Next.js dashboard but no implementation existed. A Streamlit app fills
  the frontend role immediately: no build step, runs from the existing
  Python venv, and is trivial to run headless in CI (AppTest).
- The dashboard performs no reasoning: a small httpx client
  (`frontend/dashboard/client.py`) only serializes requests, parses
  responses, and surfaces typed errors; all business logic stays in the
  backend (`api/services.py` -> CrewAI crew).
- The API contract stays the source of truth; a future Next.js
  dashboard can reuse the same endpoints.
- Client logic is tested hermetically with `httpx.MockTransport`, and
  the UI is smoke-tested with `streamlit.testing.v1.AppTest`.

---

ADR-011

Training is exposed as an API endpoint (`POST /api/v1/train`) so the
system can go from raw dataset to served predictions without a manual
CLI step, and n8n drives the whole lifecycle in a single workflow.

Reason

- To make the system functional end-to-end on a CPU-only machine, the
  gap was the missing "dataset → model" link: the API could only serve,
  so a model had to be trained by hand before anything worked.
- Training lives in the service layer (`AnalysisService.train` in
  `api/services.py`): the route only validates and delegates (ADR-009),
  preprocessing uses `preprocessing.csv.CSVPipeline`, fitting uses
  `models.TabularClassifier`, scoring uses `evaluation`, and the
  optional federated path reuses `federated.FedAvgServer` / `FederatedClient`.
- The fitted model replaces the service's model in memory, so
  `predict` / `analyze` use the new artifact immediately — no restart.
- Central fit is the default serving path (fast, deterministic). The
  federated FedAvg path is available per request (`federated: true`),
  but the federated path only supports `model_name='mlp'` (the only
  estimator with incremental `partial_fit`).
- n8n remains orchestration-only (`AGENTS.md`): the single
  `healthcare-endtoend.json` workflow triggers `train` and `analyze`,
  writes the report file, and formats the response; all reasoning stays
  in the backend crew.
- A preset registry (`PRESETS`) maps the shipped datasets
  (diabetes / heart / kidney / sepsis) to files and target columns;
  arbitrary CSVs remain supported via `dataset` + `target`.

---

ADR-012

Image analysis is exposed through the same service-boundary pattern as
tabular analysis, and the dashboard renders friendly per-feature inputs
instead of raw JSON.

Reason

- To make the dashboard usable for clinicians, JSON text areas are
  replaced with one numeric input per model feature. The feature list
  comes from the new `GET /api/v1/model` endpoint, so the form adapts
  to whatever model is served; raw JSON remains as a fallback when no
  model (and thus no feature list) is configured.
- Image uploads reuse the existing pipeline: `preprocessing.image`
  (`ImagePipeline`) → `models.image.ImageClassifier` → the crew. The
  crew (`ClinicalCrew`) gains an `image_model` / `image` branch that
  mirrors the tabular branch (prediction → risk → evidence → report),
  so image reports share the same `ClinicalReport` schema and rendering.
- `ImageClassifier` was changed to map labels by `np.unique` order
  instead of `int(label)`, so string classes (e.g. the brain-tumor MRI
  folder names) work without a manual label encoder.
- `POST /api/v1/analyze/image` takes a base64-encoded image in a JSON
  body (decoded by a Pydantic field validator) rather than a multipart
  upload, keeping the client, route, and tests uniform with the rest of
  the API. The image model is loaded lazily from `API_IMAGE_MODEL_PATH`
  like the tabular model from `API_MODEL_PATH`.
- Training the image CNN is an offline script
  (`scripts/train_image_model.py`) rather than an API endpoint, because
  CNN training is slow enough that a long-running HTTP request would be
  a poor UX; the trained artifact is then served by the API.

---

ADR-013

Privacy-preserving federated learning (paper Section 8) uses Opacus
DP-SGD for local training, a pairwise one-time-pad secure aggregator on
the server, and anonymization/pseudonymization of raw frames, with a
privacy-metrics block returned by the training API.

Reason

- The old demo's `CrewAI/app/federated/privacy.py` was the reference
  for the mechanisms (noise-multiplier DP, epsilon/delta targets,
  pairwise OTP secure aggregation, MIA-AUROC + leakage-rate metrics) and
  is ported, not rewritten, into `backend/federated/privacy.py`.
- Opacus DP-SGD requires a `torch.nn.Module`, so the federated DP path
  uses a new torch MLP (`models/csv/TorchMLPClassifier`) instead of the
  sklearn `TabularClassifier`. Its parameter exchange reuses the same
  `get_parameters` / `set_parameters` contract, so `FederatedClient`,
  `FedAvgServer`, and `average_weights` stay model-agnostic. Without
  DP (the default), the federated path keeps the sklearn `TabularClassifier`.
- Secure aggregation is a client-side additive mask that only cancels
  under equal per-client weights, so `SecureAggregator.aggregate`
  requires equal weights (matching `FedAvgServer`, which weights every
  client once). This is stricter than `average_weights` but keeps the
  masking algebra exact.
- Privacy is opt-in per training request: `POST /api/v1/train` accepts
  `differential_privacy`, `noise_multiplier`, `max_grad_norm`,
  `privacy_delta`, and `secure_aggregation`. When DP is enabled the
  response's `federated_metrics.privacy` block reports epsilon,
  budget-used %, MIA-AUROC (members = client shard rows, non-members =
  hold-out), attack-resistance score (clamped to `[0, 1]`), data
  leakage rate, and the mechanism string.
- Opacus runs with `secure_mode=False` (faster experimentation); a
  production pass must re-run with `secure_mode=True` (BACKLOG).
- Anonymization/pseudonymization helpers live in the federated privacy
  module (not preprocessing) because they exist to protect federated
  exchange payloads; raw frames never leave a client, so the API's
  leakage rate is structurally zero unless a payload path is added.
- Anonymization is actively called at both hospital data ingestion paths
  (`canonical.load_canonical_frame` and `hospitals.build_hospital_sites`)
  — not merely available as a helper. Verified: sepsis CSV's `insurance`
  column is dropped before any downstream use.
- Payload inspection (`inspect_federation_payloads`) measures data
  leakage from actual numpy arrays transmitted between clients and server,
  not hardcoded. Cumulative ε is labeled `naive_sum_upper_bound` because
  Opacus's accountant state is not persisted between rounds — this is an
  explicitly conservative upper bound, NOT tight RDP composition.
- MIA AUROC uses a confidence-based baseline (simplified); sample counts
  and protocol are recorded in privacy_metrics_summary for reproducibility.
- docs/PRIVACY_NOTES.md documents what the layer does and does not protect.

---

ADR-014

Encrypted communication is a deployment-layer concern, not an in-process
one: all inter-service traffic (API ↔ dashboard, API ↔ n8n, and any
future deployment) is secured with TLS/mTLS at the transport boundary
rather than by application-level encryption inside the framework.

Reason

- The framework runs as a local research system (FastAPI on localhost,
  Streamlit, n8n in Docker), so there is no trusted third party to
  encrypt to at the application layer, and adding one would be
  premature.
- Transport-level encryption is the industry-standard control for
  health-adjacent HTTP traffic: a reverse proxy (e.g. nginx/Caddy) or
  the deployment platform terminates TLS, and mTLS authenticates clients
  where mutual trust is required. This covers every communication path
  uniformly, including the n8n webhook and the dashboard's API client.
- The in-process data protections already implemented stay where the
  paper's threat model lives: DP-SGD (ADR-013) protects model gradients
  during federated aggregation, secure aggregation masks per-client
  updates, and PHI never leaves a client. Transport encryption
  complements these by protecting data in transit between separately
  deployed components.
- No application code change is required: the API already binds via
  uvicorn, so it can sit behind any TLS-terminating proxy unchanged.
  Enabling TLS is documented in the README "Privacy & Security" section
  rather than hardcoded, keeping the local dev loop simple.
---

ADR-015

Heterogeneous multi-disease federation maps every hospital's local columns onto
a shared canonical schema (`federated/canonical.py`) derived from the proposal's
"Expected Inputs" (11 features: age, gender, bmi, blood_pressure, heart_rate,
spo2, glucose, creatinine, cholesterol, hemoglobin, albumin) with a binary
`has_disease` target.

Reason

- The proposal requires four hospitals to own DIFFERENT disease datasets
  (Diabetes / Heart / CKD / MIMIC-IV), not partitions of one dataset.
- Standard FedAvg can only average weights when all models have identical
  shapes; per-disease adapters map local columns onto the shared schema so
  weight shapes always match.
- The `--heterogeneous` CLI flag uses each hospital's own CSV as-is —
  never partitioned or overwritten. Anonymization runs at data loading.
- Verified md5-identical across runs: hospital data files are never modified.
- Flower gRPC runs register under preset `multi_disease` in the registry;
  global model artifacts carry `feature_names` for serving.

---

ADR-016

n8n orchestrates individual agent steps via per-agent FastAPI endpoints
(`POST /api/v1/agents/{step}`), rather than proxying a single monolithic
`POST /api/v1/analyze` call.

Reason

- The proposal distinguishes n8n (workflow orchestration) from CrewAI
  (agentic reasoning) from FastAPI (service boundary). Blurring them by
  having n8n proxy one big call undermines the architectural claim.
- `clinical-full-v2.json` chains: Receive → Validate → Patient Analyst →
  Disease Predictor → Build RAG Query → Medical Researcher →
  Treatment Planner → Explainability Expert → Assemble Report →
  Validate → Store → IF High Risk → Doctor Notify → Respond.
- Each HTTP node calls a DIFFERENT `/api/v1/agents/*` endpoint, so n8n
  visibly orchestrates the pipeline while CrewAI reasoning stays inside
  each API call.
- The deterministic path (`run_analysis()`) also executes seven discrete
  agent steps with full per-agent tracing (`CrewTrace` / `AgentTrace`),
  satisfying M4 Definition of Done without requiring an LLM key.

---

ADR-017

Longitudinal risk monitoring (`backend/risk/`) and clinician feedback with
threshold-gated retraining (`backend/feedback/`) are standalone modules that
persist to SQLite alongside the federation registry.

Reason

- Every `analyze()` call persists risk score, level, prediction, confidence,
  and markers to `artifacts/risk_history.db`, enabling longitudinal trend
  analysis and escalation alerts (score jump > threshold between visits).
- Clinician feedback labels accumulate in `artifacts/feedback.db`; once
  pending count ≥ threshold, n8n triggers retraining on the augmented
  dataset (base + feedback rows) and serves the new model immediately.
- Both stores use `sqlite3.connect(check_same_thread=False)` because FastAPI
  handles requests on multiple threads.
- n8n `risk-monitoring.json` polls `/api/v1/risk/alerts` every 15 minutes
  and fires clinician notifications per active escalation alert.

---

ADR-018

Secondary-outcome heads (P2.1): the sepsis disease model no longer trains on
`readmission_30day` (target leakage — a future outcome column was a
feature), and a separate `readmission_30day` head trains via
`POST /api/v1/train {"preset": "sepsis", "outcome": "readmission_30day"}`.

Reason

- A diagnosis model must only see admission-time features. `readmission_30day`
  is known only after discharge, so training on it leaks the future; the
  sepsis feature count drops 75 → 74 and the baseline study was regenerated.
- The head trains on the same admission-time vector (fellow secondaries and
  the disease target are also excluded), so both models share an identical
  feature space and `analyze()` scores both from one row.
- Honest numbers on the real label (central logistic): accuracy 0.946 but
  F1 0.486 / ROC-AUC 0.500 — the 5.4%-positive label means accuracy is
  mostly the majority class; the head has no ranking signal yet.
- Mortality stays un-trainable: no shipped dataset has a mortality column
  (verified at header level). It is blocked on a credentialed MIMIC-IV
  extract (P2.6 track), not on modelling — the dashboard says so instead of
  "Not estimated".

---

ADR-019

Model-derived explanations (P2.2): the Explainability Expert no longer
reports magnitude-sort as the explanation. Tabular drivers are SHAP values
from the fitted estimator — exact `LinearExplainer` for logistic
regression, bounded `KernelExplainer` otherwise — and the image CNN
explains via Grad-CAM from the last convolutional layer. Magnitude-sort
survives only as an explicitly labeled heuristic fallback.

Reason

- "Largest raw value drives the prediction" is not an explanation of the
  model; SHAP attributes the fitted model's own disease-class output, so
  the text names what actually pushed the prediction. Class `1` is always
  the disease class, hence positive SHAP means toward disease.
- SHAP needs an honest reference distribution, so `train()` persists a
  stratified ≤25-row preprocessed background per preset
  (`artifacts/<preset>/shap_background.joblib`); the served background
  only applies when its preset matches the active one, otherwise the
  fallback says it is not model-derived instead of silently mixing
  distributions.
- KernelExplainer cost scales with background size and feature count, so
  both are capped (25 rows, ≤500 evals) — a single-row explanation stays
  near ~1s on CPU.
- Grad-CAM needs no extra dependency (native torch hooks) and returns a
  normalized [0, 1] heatmap plus a peak-quadrant/coverage summary line.

---

ADR-020

Evidence-grounded, model-graded treatments (P2.3): playbook candidates
are no longer returned in static order with a single "source consulted"
pointer. `build_treatment_recommendations` now accepts retrieved
`evidence` and model-derived `drivers`, scores each candidate by
content-token overlap with evidence chunks plus driver-mention
relevance, and returns ranked recommendations with inline `[evidence:
doc_id]` citations — or an explicit `[playbook-only: no retrieved
evidence]` label when no chunk substantively supports a candidate.

Reason

- A static playbook cannot claim to be evidence-based; citations must
  attach per recommendation, and unsupported advice must say so rather
  than borrow credibility from an unrelated "source consulted" line.
- Ranking toward the patient's SHAP drivers (Agent 4 computes the
  attribution once and shares it with Agent 5) makes the plan lead with
  what drove this patient's prediction instead of generic order.
- Unsupported candidates are kept, not dropped: an empty plan is worse
  than a labeled playbook-only one, and the label keeps the honesty
  framing. No LLM, no new dependency — deterministic token overlap.
- Callers with no evidence or drivers (and the
  `/agents/treatment-planner` fallback path) keep the exact legacy
  strings, so existing reports are unaffected.

---

ADR-021

Disease-aware n8n RAG query (P2.4): the "Build RAG Query" node no
longer emits hardcoded `diabetes treatment` / `healthy lifestyle` —
it anchors on the Disease Predictor's `disease` output, mirroring the
backend `build_disease_query` wording (positive: diagnosis/management
guidelines; negative: prevention/screening), with generic branches
when no disease is exposed.

Reason

- The hardcode was doubly broken: it named diabetes for every patient,
  and its output was silently ignored — `/agents/evidence-retrieval`
  rebuilt its own feature-only query and `AgentStepRequest` had no
  `query` field at all.
- The route now prefers a caller-supplied `query` (falling back to the
  marker-anchored builder on blank), and `/agents/disease-predictor`
  exposes `disease` + `predicted_label` enriched from the service's
  active preset, so the workflow chain carries real clinical context
  instead of a class integer.
- Verified end to end: the node's JS executes under node against six
  predictor shapes (heart/kidney/diabetes ±, unlabeled, bare), and
  contract tests pin the node wording, the predictor→builder→researcher
  wiring, and the query pass-through.

---

ADR-022

Single-agent LLM polish over a seven-stage deterministic pipeline
(P2.5): the "multi-agent" claim is re-scoped, not restored. The
optional LLM layer is one report-writer agent, one task, one kickoff
call, merged over the deterministic base report with predictions, risk
scores, and evidence preserved verbatim. The seven deterministic stages
(Patient Analyst → Disease Predictor → Medical Researcher → Treatment
Planner → Explainability Expert → Risk Monitor → Report Writer) each
emit an `AgentTrace` into a `CrewTrace`, and honest agent metrics are
computed from those traces.

Reason

- A genuine five-agent LLM chain needs an API key on every run, costs
  ~5x per analysis in latency and quota, and reintroduces the safety
  regression P2.2/P2.3 closed: an LLM re-deriving numbers instead of
  polishing them. The deterministic pipeline already returns verified
  values without a key; the LLM adds narrative only.
- Each stage has a single responsibility and a recorded
  input/output/status/timing trace, which is what the paper's RQ — "Can
  Multi-Agent AI outperform a single-agent architecture?" — can actually
  measure: per-stage completion, consistency, and collaboration from
  real traces versus a single-agent baseline. A single LLM pass with no
  staging offers nothing to ablate.
- What changed: the proposal §6 "Multi-Agent Architecture" lists one
  narrative-polish agent instead of five reasoning agents; README and
  SOFTWARE_ARCHITECTURE describe the 7-stage traced pipeline plus the
  optional single-call polish; `agents.py`/`config.py` comments no longer
  reference a lean five-agent crew; `test_run_analysis_traces_seven_agents_in_order`
  pins the stage names and order so any future collapse or rename
  forces this ADR to be revisited.
