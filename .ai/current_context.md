# Current Context

## Current Milestone

Repository cleanup: removed all tests, examples, scripts, dead
multimodal modules, `docs/`, and root junk. The live system (FastAPI
backend + CrewAI orchestrator + n8n + Streamlit dashboard) is unchanged
and verified to import cleanly.

## Current Module

N/A (cleanup session).

## Current Task

1. Delete all test files (backend + frontend) and `backend/conftest.py`.
2. Delete `backend/examples/`, `backend/scripts/`, and top-level
   `scripts/` (run_system.sh, train_image_model.py).
3. Delete dead multimodal modules (`backend/models/multimodal/`,
   `backend/preprocessing/multimodal/`) and remove their references from
   `models/__init__.py` / `preprocessing/__init__.py`.
4. Delete `docs/`, `n8n/README.md`, and root junk (`workflow.txt`,
   `ai-automation-research.md`, empty log files, caches).
5. Consolidate all documentation into `README.md` with an accurate
   Mermaid flowchart of the actual system.
6. Update `AGENTS.md` and `.ai/` to match the cleaned repo.

## Completed

- All tests (~38 tracked files, ~6k lines) removed.
- `backend/examples/`, `backend/scripts/`, top-level `scripts/` removed.
- Multimodal modules removed; `models/__init__.py` and
  `preprocessing/__init__.py` updated.
- `docs/` (10 files), `n8n/README.md`, `workflow.txt`,
  `ai-automation-research.md`, empty logs, and caches removed.
- Backend + frontend verified: all live imports resolve, code compiles.
- `AGENTS.md` de-referenced from deleted docs/scripts/tests.

## Next Files (optional / backlog)

- Write the consolidated `README.md` (accurate Mermaid flowchart,
  quick-start, API reference, config).
- Update `.ai/next_session.md`.
- Commit the cleanup as one focused commit when the user asks.

## Design Notes

- Tests were removed by explicit user request. Verification of the live
  system is the import smoke check + manual API calls in README.md.
- The remaining code is the genuinely working system: preprocessing →
  models → federated → RAG → CrewAI → FastAPI → n8n → dashboard.

## Status

Cleanup staged (deletions staged in git index, `models/__init__.py` and
`preprocessing/__init__.py` modified, README not yet rewritten).