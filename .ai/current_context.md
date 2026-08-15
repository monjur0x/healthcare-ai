# Current Context

## Current Milestone

Milestone 10 — evaluation-gap closure (paper §12) (complete, committed)

## Current Module

rag/ · CrewAI/orchestrator/ · federated/ · docs/ · .ai/

## Current Task

Closed the remaining gaps against the research proposal with six focused
commits (all pushed to `origin/main`):

1. `fix(federated)` — declare `opacus>=1.5.0` in `backend/requirements.txt`
2. `feat(rag)` — ChromaDB persistent vector store (`ChromaVectorStore` +
   `build_vector_store()` factory, `RAG_VECTOR_STORE`)
3. `feat(rag)` — `SentenceTransformerEmbedder` (opt-in dense embeddings,
   default `BAAI/bge-small-en-v1.5`)
4. `feat(rag)` — RAGAS-style metrics (`context_precision` /
   `context_recall` / `faithfulness` / `answer_relevancy` +
   `RAGQualityMetrics`)
5. `feat(crew)` — agent metrics (`task_completion_rate` /
   `decision_consistency` / `agent_collaboration_score` + `AgentMetrics`
   wired into `ClinicalReport.agent_metrics`)
6. `docs(security)` — ADR-014 (transport-layer TLS/mTLS) + README
   "Privacy & Security"
7. `style(rag)` — black/ruff/isort cleanup + test-file rename

Backend suite **326 passing** (+37), frontend **13 passing**, lint clean.
TF-IDF / in-memory store stay the defaults; Chroma and sentence-transformers
are opt-in. `opacus` declared so the DP path is reproducible.

Open question still pending: `gemini-3.7-flash` vs `gemini-3.6-flash` as
the CrewAI default LLM.

## Completed

- Milestones 1–9 (preprocessing, models, evaluation, federated, RAG,
  CrewAI orchestration, FastAPI, n8n, Streamlit, end-to-end system,
  image analysis, privacy-preserving federated learning) — all
  committed and pushed.
- Milestone 10 — evaluation-gap closure (this session):
  - `backend/rag/store_chroma.py` — `ChromaVectorStore` (persistent
    ChromaDB, cosine-only, `EmptyCorpusError` on empty search, same
    `add` / `search` / `__len__` interface); `store.py::build_vector_store()`
    factory; `RAGSettings` gains `VECTOR_STORE` / `CHROMA_PERSIST_DIR` /
    `CHROMA_COLLECTION`; exported from `rag/__init__.py`
  - `backend/rag/embedder.py` — `SentenceTransformerEmbedder` (dense,
    lazy model load, BGE query-instruction prefix, `EmbeddingError` when
    the dependency is missing); `build_embedder("sentence-transformer")`;
    `SENTENCE_TRANSFORMER_MODEL` setting; `RAG_EMBEDDING_MODEL`
    documented
  - `backend/rag/metrics.py` — `context_precision`, `context_recall`,
    `faithfulness`, `answer_relevancy`, `RAGQualityMetrics` +
    `rag_quality_metrics()` aggregator (all LLM-free, embedder-agnostic)
  - `backend/CrewAI/orchestrator/metrics.py` — `task_completion_rate`,
    `decision_consistency`, `agent_collaboration_score`,
    `AgentMetrics` + `compute_agent_metrics()`; `ClinicalReport` gains
    optional `agent_metrics`; wired through `assemble_clinical_report`
  - Dependencies declared: `chromadb>=0.5.0`,
    `sentence-transformers>=3.0.0` (RAG), `opacus>=1.5.0` (DP)
  - README Configuration + `backend/.env.example` document all new RAG_
    variables; README gains "Privacy & Security" section
  - ADR-014 in `docs/DECISIONS.md`
  - Lint tools installed into the CrewAI venv (black/ruff/isort) since
    they were missing from the environment
  - Tests: backend 326 (+37), frontend 13; lint clean (black / ruff /
    isort, run from `backend/` so `backend/pyproject.toml` covers the
    frontend too)

## Next Files (backend)

- Real flwr `run_simulation` / networked `ServerApp` (blocked: `ray`
  not installed)
- Production DP pass: re-run with Opacus `secure_mode=True` (currently
  `secure_mode=False` for experimentation speed — UserWarning asks for a
  final secure retrain before release)
- Orchestrator LLM path: settle `gemini-3.7-flash` (503 overloaded) vs
  `gemini-3.6-flash` default
- API hardening: full OAuth, file-upload endpoint, deployment container,
  downstream n8n storage/notification branches
- Wire RAGAS / agent metrics into an API response or standalone
  evaluation endpoint (currently library functions + tests only)
- RAGAS-vs-heuristic calibration (LLM-free proxies vs a judge-LLM
  baseline on a labeled set)
- Baseline comparison study (paper §13)

## Design Notes

- ADR-007: TF-IDF + in-memory store are the default RAG path; dense
  embeddings (sentence-transformers) and persistent backends (ChromaDB)
  are opt-in behind the `Embedder` / `VectorStore` interfaces (Milestone
  10 added `SentenceTransformerEmbedder` + `ChromaVectorStore` + the
  `build_vector_store()` factory).
- ADR-008/009/010/011/012/013 — unchanged; see prior session notes.
- ADR-014: encrypted communication is a deployment-layer concern — all
  inter-service traffic (dashboard ↔ API, n8n ↔ API) uses TLS/mTLS at a
  reverse proxy; in-process protections (DP-SGD, SecAgg, PHI-never-leaves)
  remain the paper's threat model. No application code change required.
- `rag/tests/test_vector_store_chroma.py` must keep temp-dir isolation:
  ChromaDB ties at identical cosine are broken non-deterministically
  (unlike the in-memory NumPy store), so tests avoid exact-tie asserts.
- CrewAI venv (`backend/CrewAI/.venv-opencode`) has crewai 1.15.11,
  pydantic 2.12, qdrant-client, sentence-transformers 5.6.1, chromadb
  1.1.1, flwr, torch, fastapi 0.138, uvicorn, httpx, streamlit 1.61,
  opacus 1.6.0, plus black / ruff / isort.
- Lint/test commands: run from `backend/` — `ruff check . ../frontend`,
  `black --check . ../frontend`, `isort --check-only . ../frontend`
  (linters are now in the CrewAI venv). Do NOT run ruff from the repo
  root (defaults apply to `frontend/`, no config found) and do NOT use
  `--config backend/pyproject.toml` from root (src-relative first-party
  detection mis-orders backend imports).
- Test basenames must be unique across `backend/`: rename happened for
  `CrewAI/orchestrator/tests/test_metrics.py` →
  `test_agent_metrics.py` (collided with `evaluation/tests/test_metrics.py`).

## Status

Milestones 1–10 complete and pushed to `origin/main`. Backend 326 +
frontend 13 tests passing, lint clean. The six-gap closure against the
research proposal (opacus dep, ChromaDB, sentence-transformers, RAGAS
metrics, agent metrics, ADR-014) is done. Open question: `gemini-3.7-flash`
(default) returns transient 503s; user may prefer `gemini-3.6-flash`.
Next: real flwr deployment path, then API hardening / metric exposure.