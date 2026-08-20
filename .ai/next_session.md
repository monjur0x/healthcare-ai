# Next Session

## Objective

Continue the multi-hospital federated build. Phase 1 (distributed Flower
deployment), the Phase 2 federation registry API, the dashboard Federation
panel, and the real medical RAG corpus are complete and verified. Next:
commit the RAG corpus work and pick the next backlog item.

## Done This Session

- Authored the bundled medical corpus at `backend/rag/corpus/` (7
  documents covering diabetes, hypertension, chronic kidney disease,
  sepsis, coronary heart disease, obesity/metabolic health, and clinical
  laboratory values).
- Added `backend/rag/corpus.py` with `load_documents` /
  `load_bundled_corpus` (recursive discovery, deterministic ordering,
  source metadata from file name); exported from `rag/__init__.py`.
- Refactored `build_rag_pipeline` (services.py) to ingest the bundled
  corpus by default, keeping the legacy `DEFAULT_CORPUS` as a fallback.
- Verified against the live server: 7 documents → 8 TF-IDF chunks;
  retrieval returns relevant evidence for metformin, septic shock,
  eGFR/CKD, and LDL targets.
- Restarted the live API; `/api/v1/retrieve` serves real corpus evidence.
- Updated README, `.env.example`, and `backend/api/config.py` docstring.

## Next Steps

1. Commit + push the RAG corpus work when the user asks.
2. Phase 2+ candidates (backlog):
   - Doctor notification (n8n alert when risk is high).
   - Feedback loop (n8n → retrain → redeploy on threshold).
   - Encrypted gRPC transport (Flower certificates).
   - Risk monitoring over persisted patient history.
   - Cross-host deployment of hospital clients (documented commands).
   - Expand the corpus with more conditions / treatment guidelines.

## Open Questions

- Commit cadence for the RAG corpus work.