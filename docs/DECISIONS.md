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