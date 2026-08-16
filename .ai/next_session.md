# Next Session

## Objective

The baseline comparison study (paper §13) is complete, committed, and
pushed: `backend/scripts/baseline_study.py` + `docs/BASELINE_STUDY_RESULTS.md`
(real numbers for all four datasets + Findings for RQ1–RQ4). Remaining work
is the optional backlog / the earlier evaluation-gap discussion.

## Done This Session (no further action)

- Baseline study implemented, tested, run on the real datasets, and
  committed as `feat(eval): add baseline comparison study (paper §13)`.
- Fixed `prepare_tabular_data` to strip whitespace from string labels
  (`'ckd\t'` == `'ckd'`) — the kidney preset previously failed the
  federated partition (phantom third class with < 3 samples).
- Headline results (shared split, test_size=0.25, seed=42; 3 clients × 5
  rounds): federated Δ accuracy +0.027 (diabetes) / +0.026 (heart) /
  +0.020 (kidney) / 0.000 (sepsis) vs centralized; RAG context recall
  1.000 with precision 0.300–0.350 at top-k=5; agent completion 0.6→0.8 and
  collaboration 0.4→0.6 with RAG evidence wired into the deterministic crew.
- Backend suite 335 passing (+4 script tests), lint clean.

## Optional Next Steps

1. Backlog (unchanged candidates, pick a direction):
   - Patient persistence + history in the dashboard
   - Backend mortality/readmission risk models (replace "not estimated")
   - Model-derived / SHAP explainability for the decision report
   - Qdrant vector-store backend (paper §13 context: the RAG evaluation in
     the baseline study showed low precision at top-k=5 — the dense
     `SentenceTransformerEmbedder` is the cheapest lever, Qdrant a storage
     one)
   - Open-source / local LLM provider for the crew (`transformers` is
     installed; `CrewSettings.LLM_PROVIDER` currently only has `google`)
2. If the baseline study is re-run, the Findings section is preserved
   automatically (script keeps everything from `## Findings` onward).
3. Run tests + lint from `backend/` (`pytest ... scripts/tests`, `ruff
   check . ../frontend`); never ruff from the repo root.

## Open Questions

- `gemini-3.7-flash` still returns transient 503 "high demand";
  `gemini-3.6-flash` verified stable — consider switching
  `CrewSettings.LLM_MODEL` default.
