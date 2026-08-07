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