# Next Session

## Objective

Finish the repository cleanup: write the consolidated `README.md` with
an accurate Mermaid flowchart of the live system, then commit the
cleanup.

## Done This Session

- Deleted all test files (backend + frontend, ~38 tracked files) and
  `backend/conftest.py`.
- Deleted `backend/examples/`, `backend/scripts/`, top-level `scripts/`.
- Deleted dead multimodal modules (`backend/models/multimodal/`,
  `backend/preprocessing/multimodal/`) and fixed `models/__init__.py` /
  `preprocessing/__init__.py`.
- Deleted `docs/`, `n8n/README.md`, root junk, empty logs, caches.
- Verified the live backend + frontend import cleanly.
- Updated `AGENTS.md` and `.ai/current_context.md` to match.

## Next Steps

1. Rewrite `README.md` — accurate system flowchart (Mermaid), quick
   start, API endpoints, env config, n8n workflow docs.
2. Commit the cleanup as one focused commit (user's call).
3. Backlog candidates (from previous sessions):
   - Patient persistence + history in the dashboard
   - Mortality/readmission risk models
   - SHAP-style explainability
   - Encoder (categorical level map) persistence
   - Faster LLM model for the crew (kickoff is ~6 min)

## Open Questions

- Whether to commit the cleanup now (user decides).