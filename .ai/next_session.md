# Next Session

## Objective

Milestone 6 — n8n orchestration workflows, then the real flwr deployment
path.

## Suggested Steps

1. Read `docs/SOFTWARE_ARCHITECTURE.md`, `docs/DECISIONS.md`, and
   `.ai/current_context.md`.
2. Read `docs/BACKLOG.md` for the scoped items.
3. Pick one of:
   - n8n: workflow definitions that trigger the API (`/api/v1/analyze`,
     `/api/v1/predict`, `/api/v1/retrieve`) / the crew; orchestration
     only — AI reasoning stays in CrewAI (`AGENTS.md`).
   - Real flwr driver: wrap the synchronous `FedAvgServer` round logic
     in a `ServerApp`/`ClientApp` and run `flwr.simulation.run_simulation`
     (Ray backend). BLOCKED until `ray` is installed (heavy dependency).
   - API hardening: file-upload endpoint for CSV / image inference,
     full OAuth (currently optional static `API_TOKEN`), deployment
     Dockerfile for `uvicorn api.main:app`.
   - Port differential privacy from the removed old demo
     (`backend/CrewAI/app/federated/privacy.py`, noise-multiplier
     approach) into `federated/` — see `docs/BACKLOG.md`.
4. Add unit tests under the module's `tests/` directory; keep test-file
   basenames unique across `backend/`.
5. Run: `pytest preprocessing/tests models/tests evaluation/tests federated/tests rag/tests examples/tests CrewAI/orchestrator/tests api/tests`,
   black, ruff, isort (against `backend/pyproject.toml`); use the CrewAI
   venv (`backend/CrewAI/.venv-opencode`, has flwr 1.33.0, torch,
   sklearn, PIL, pydantic-settings, crewai 1.15.11, fastapi 0.138).
6. Update `docs/DEVELOPMENT_STATUS.md`, `docs/CHANGELOG.md`,
   `docs/BACKLOG.md`, `docs/DECISIONS.md` (if a new ADR),
   `.ai/current_context.md`, `.ai/next_session.md`.

## Conventions Reminder

- Preprocessing stays in `preprocessing/`, models in `models/`,
  retrieval in `rag/`, orchestration in `CrewAI/orchestrator/`,
  API in `api/`.
- API routes only validate + delegate to `api/services.AnalysisService`;
  domain exceptions are mapped to `APIError` subclasses (ADR-009).
- Federation only moves weights; training/inference stay in models.
- Agents orchestrate reasoning and consume pipeline outputs; they never
  implement ML algorithms (see `AGENTS.md`).
- CrewAI agents/tasks/crew construct hermetically (no LLM key); the LLM
  path (`ClinicalCrew.run_llm`) needs `CREW_LLM_API_KEY` and a provider
  extra installed.
- `get_parameters` returns memory-shared NumPy views of the model
  state; snapshot with `.copy()` before the model trains further.
- Log via `get_logger(__name__)`; raise module exceptions from the
  module's `exceptions.py`.

## Open Questions

- Whether the real `run_simulation` integration check belongs in the
  unit suite (Ray spawn is slow/flaky) or as a separate smoke script.
- Whether the RAG production path should adopt sentence-transformers /
  Qdrant (ADR-007 defers this behind the `Embedder` / `VectorStore`
  interfaces).
- Which LLM provider to wire into `ClinicalCrew.run_llm`
  (needs an API key; never commit secrets).
- n8n: whether to commit plain JSON workflow definitions (n8n format)
  or generate them from a small Python config.