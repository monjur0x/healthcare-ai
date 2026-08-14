# Current Context

## Current Milestone

Milestone 3 — RAG (complete)

## Current Module

backend/rag

## Current Task

The RAG module (`backend/rag/`) and its demo are complete and pushed.
Next: CrewAI agents consuming preprocessing / model / RAG outputs, then
`api/` and `n8n/`.

## Completed

- Milestone 1: preprocessing (CSV + image + multimodal), 70 tests
- Milestone 2: models, evaluation, federated tie-in, sync FedAvg server,
  CSV → FedAvg demo, CNN federation, federated metrics, image FedAvg
  demo (see git history and `docs/DEVELOPMENT_STATUS.md`)
- `backend/rag/` — document ingestion → chunking → embedding → vector
  search → context retrieval:
  - `exceptions.py` — `RAGError` + `EmptyCorpusError`,
    `EmptyQueryError`, `InvalidDocumentError`, `EmbeddingError`,
    `RetrievalError`
  - `config.py` — `RAGSettings` (env prefix `RAG_`): chunk size/overlap,
    embedding model, max features, top-k, similarity metric
  - `documents.py` — frozen `Document` / `Chunk` / `RetrievalResult`
    dataclasses with `to_dict()`; `Chunk` inherits the document source
  - `chunker.py` — `TextChunker`: deterministic word-based sliding
    window with overlap
  - `embedder.py` — `Embedder` ABC, `TfidfEmbedder` (corpus-fitted,
    default), `HashingEmbedder` (fit-free fixed-dim), `build_embedder`
  - `store.py` — `VectorStore`: in-memory NumPy nearest-neighbour over
    cosine / dot
  - `retriever.py` — `Retriever`: incremental ingest, query → top-k
    chunks, `build_context` (source-labelled prompt block)
  - `metrics.py` — `precision_at_k`, `recall_at_k`,
    `mean_reciprocal_rank`, `RetrievalMetrics`
  - `pipeline.py` — `RAGPipeline` composing chunker → embedder → store →
    retriever (`ingest_documents`, `ingest_texts`, `retrieve`,
    `build_context`)
  - No new dependencies (reuses scikit-learn); ADR-007
- `examples/rag_demo.py` — corpus directory → `RAGPipeline` → queries →
  top-k chunks + context; quality metrics when `--ground-truth` JSON
  (query → relevant document ids) is supplied; writes `report.json`
- Tests: RAG 38 + examples 2 (demo smoke tests) new; full suite
  **195 passing** (`pytest preprocessing/tests models/tests evaluation/tests federated/tests examples/tests rag/tests`)
  — black / isort / ruff clean

## Next Files (backend)

- `CrewAI/` agents/tasks consuming preprocessing + model + RAG outputs
  (retrieval tools wrapping `RAGPipeline`)
- `api/` FastAPI routes (services only; no business logic in routes)
- `n8n/` orchestration triggers
- `federated/` — real flwr `run_simulation` / networked `ServerApp`
  (blocked: `ray` not installed); privacy budget metrics

## Design Notes

- RAG follows the preprocessing pipeline pattern: `RAGPipeline` is the
  single reusable entry point for CrewAI / API / examples.
- Embedding and storage are swappable behind `Embedder` ABC and
  `VectorStore`; dense models (sentence-transformers) and Qdrant are the
  deferred production path (ADR-007).
- `Retriever.build_context` labels chunks with document id + source —
  the prompt-ready context block for CrewAI tasks.
- Demo report shape: corpus stats + per-query results (document ids +
  scores) + context + optional metrics.
- Testing: use the CrewAI venv (`backend/CrewAI/.venv-opencode`).
- Existing `CrewAI/app/models/*` are old demos; do not mix with
  `backend/models/`.

## Status

Milestone 3 (RAG module + demo) complete and pushed. Next milestone is
CrewAI agent orchestration, then FastAPI, then n8n.