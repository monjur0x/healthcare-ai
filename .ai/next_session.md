# Next Session

## Objective

Commit the live n8n verification fixes (working tree is dirty) as focused
commits, then push. Read `docs/DEVELOPMENT_STATUS.md` +
`.ai/current_context.md` first (AGENTS.md workflow).

## Suggested Steps

1. Review `git diff` for:
   - `n8n/healthcare-endtoend.json` — `$json.body.*` field references,
     removed Write-to-disk node, `firstIncomingItem` respond nodes
   - `n8n/clinical-analysis.json` — `firstIncomingItem` respond nodes
   - `n8n/README.md` — corrected workflow description, credential
     requirement, Docker networking, activation note
   - docs: `CHANGELOG.md`, `DEVELOPMENT_STATUS.md`, `BACKLOG.md`,
     `.ai/*`
2. Commit as focused commits, e.g.:
   - `fix(n8n): read webhook payload from body in end-to-end workflow`
   - `fix(n8n): return report via respond node instead of disk write`
   - `docs(n8n): document credential, Docker networking, activation`
3. Push to `main`.
4. (Optional, for the user's own instance) Bring up the original
   `healthcare-n8n` (:5678): create the Header Auth credential, wire it
   to the HTTP nodes, patch backend URLs to `http://172.17.0.1:8000`
   (Docker bridge) or run n8n with host networking, import + activate the
   fixed workflow, and re-run the smoke test.
5. Pick the next backlog direction (unchanged candidates):
   - Patient persistence + history in the dashboard
   - Backend mortality/readmission risk models (replace "not estimated")
   - Model-derived / SHAP explainability for the decision report
   - Report disk-archival from n8n via a volume under the file-sandbox
     base (`~/.n8n-files`), or OAuth / deployment container
6. Run tests + lint: frontend from `frontend/` (`pytest dashboard/tests
   -q`, `ruff check streamlit_app.py dashboard/`, `black --check`,
   `isort --check-only`); backend from `backend/`. Never ruff from the
   repo root.
7. Update docs per AGENTS.md.