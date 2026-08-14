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