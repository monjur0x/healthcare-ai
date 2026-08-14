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