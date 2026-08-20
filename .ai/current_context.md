# Current Context

## Current Milestone

Phase 2 of the multi-hospital federated build: surface the federation
model registry through the FastAPI and the Streamlit dashboard, and
replace the placeholder RAG corpus with a real medical knowledge corpus
bundled in the repository.

## Current Module

`backend/rag/` (corpus loader + bundled documents) and
`backend/api/services.py` (`build_rag_pipeline`).

## Current Task

Real medical RAG corpora (complete):

1. Bundled corpus at `backend/rag/corpus/` covering the four supported
   conditions plus general clinical topics:
   - `diabetes-mellitus.md` — diagnosis, management, monitoring,
     complications.
   - `hypertension.md` — classification, risk, treatment.
   - `chronic-kidney-disease.md` — stages, markers, management.
   - `sepsis.md` — Sepsis-3, recognition, management.
   - `coronary-heart-disease.md` — risk, lipids, diagnosis, management.
   - `obesity-metabolic-health.md` — BMI, nutrition, activity.
   - `clinical-laboratory-values.md` — reference ranges for the presets.
2. `backend/rag/corpus.py` — `load_documents` / `load_bundled_corpus`
   loaders that turn the directory into `Document` objects with source
   metadata; exported from `rag/__init__.py`.
3. `build_rag_pipeline` ingests the bundled corpus by default and keeps
   the legacy `DEFAULT_CORPUS` as a last-resort fallback.
4. README + `.env.example` + config docstring updated.

## Completed

- Phase 1 (committed `ebad74f` + `51d479d`): distributed Flower gRPC
  deployment, hospitals-as-processes, SQLite registry, DP + secure
  aggregation, canonical feature schema.
- Federation registry API (committed `7afa18e`).
- Dashboard federation panel (committed `b351e0e`).
- Bundled medical corpus authored (7 documents, ~2,750 words) and verified:
  - `load_bundled_corpus` returns all 7 documents with source metadata;
  - `build_rag_pipeline` ingests 8 chunks (TF-IDF, 933 features);
  - retrieval returns relevant documents for metformin, septic shock,
    eGFR/CKD, and LDL target queries;
  - live API restarted and `/api/v1/retrieve` serves the real corpus.

## Next Files (optional / backlog)

- Doctor notification via n8n.
- Feedback / retrain loop.
- Encrypted gRPC transport.
- Risk monitoring on history.
- Cross-host deployment docs.

## Design Notes

- The corpus is bundled in the repository so the API serves real evidence
  out of the box (no external download). Users can override with
  `API_CORPUS_DIR`.
- `load_documents` scans recursively, sorts by path for determinism, and
  sets `source` to the file name so evidence items carry provenance.
- `build_rag_pipeline` keeps the legacy `DEFAULT_CORPUS` only as a
  fallback if the bundled directory is ever missing.

## Status

Real medical RAG corpora complete and verified against the live server.
Repo not yet committed (user drives commits).