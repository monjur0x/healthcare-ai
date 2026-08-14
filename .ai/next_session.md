# Next Session

## Objective

Milestone 5+ — FastAPI (`api/`), then n8n orchestration, then the real
flwr deployment path.

## Suggested Steps

1. Read `docs/SOFTWARE_ARCHITECTURE.md`, `docs/DECISIONS.md`, and
   `.ai/current_context.md`.
2. Read `docs/BACKLOG.md` for the scoped items.
3. Pick one of:
   - FastAPI `api/`: a `services/` layer that runs `ClinicalCrew` /
     `RAGPipeline` / models; routes only validate + delegate (no
     business logic in routes per `AGENTS.md`). Endpoints: prediction,
     evidence retrieval, clinical analysis report.
   - n8n: workflow definitions that trigger the API / crew.
   - Real flwr driver: wrap the synchronous `FedAvgServer` round logic
     in a `ServerApp`/`ClientApp` and run `flwr.simulation.run_simulation`
     (Ray backend). BLOCKED until `ray` is installed (heavy dependency).
4. Add unit tests under the module's `tests/` directory; keep test-file
   basenames unique across `backend/`.
5. Run: `pytest preprocessing/tests models/tests evaluation/tests federated/tests rag/tests examples/tests CrewAI/orchestrator/tests`,
   black, ruff, isort (against `backend/pyproject.toml`); use the CrewAI
   venv (`backend/CrewAI/.venv-opencode`, has flwr 1.33.0, torch,
   sklearn, PIL, pydantic-settings, crewai 1.15.11).
6. Update `docs/DEVELOPMENT_STATUS.md`, `docs/CHANGELOG.md`,
   `docs/BACKLOG.md`, `docs/DECISIONS.md` (if a new ADR),
   `.ai/current_context.md`, `.ai/next_session.md`.

## Conventions Reminder

- Preprocessing stays in `preprocessing/`, models in `models/`,
  retrieval in `rag/`, orchestration in `CrewAI/orchestrator/`,
  API in `api/`.
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
- FastAPI dependency availability in the CrewAI venv (add `fastapi` /
  `uvicorn` to `backend/requirements.txt`).