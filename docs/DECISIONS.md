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