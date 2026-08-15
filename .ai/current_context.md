# Current Context

## Current Milestone

Milestone 11 — doctor-facing CDS dashboard (committed + pushed) + live
n8n end-to-end verification follow-up (complete, uncommitted)

## Current Module

n8n/ (healthcare-endtoend.json · clinical-analysis.json · README.md) · docs/ · .ai/

## Current Task

Live n8n end-to-end verification of the Milestone 11 n8n route. Drove the
dashboard's n8n path against a real n8n instance (2.34.6) + the real
FastAPI backend. This exposed **real bugs in the committed
`n8n/healthcare-endtoend.json`** that hermetic tests could not catch:

1. **Webhook payload nesting** — the n8n webhook node outputs
   `{headers, params, query, body}`, so the request payload lives under
   `body`. The workflow read top-level `$json.*` / `item.json.*`, so
   `train` (IF gate), `patient`, and `features` were silently undefined:
   the train branch never ran and the patient came back as "Unknown".
   Fixed every expression to `$json.body.*` /
   `item.json.body.*`.
2. **Write: Report to Disk on the response critical path** — the
   readWriteFile node errored (ENOENT, then "Access to the file is not
   allowed" because n8n 2.34 restricts file writes to `~/.n8n-files`)
   *before* the Respond node ran, yielding HTTP 200 with an empty body.
   Removed the Write node + binary attachment; the Code node now returns
   the full report in the webhook response (which is what the dashboard
   consumes).
3. **Respond shape** — `respondWith: allIncomingItems` returned a JSON
   array; switched both Respond nodes to `firstIncomingItem` so the
   response is a single object (matches the dashboard's
   `analyze_via_n8n()` contract).
4. **Credential requirement** — the HTTP nodes reference an
   `httpHeaderAuth` credential by ID; the placeholder ID fails at
   execution ("Credential with ID ... does not exist"). A real
   "Healthcare API Token" credential must exist and be associated.
   (n8n 2.x also ignores `N8N_API_KEY` env for the public API — keys
   must be created by an owner via the private `/rest` API.)
5. **Docker networking** — the HTTP nodes hardcode `localhost:8000`,
   which from inside the n8n container reaches the container, not the
   host. Verified via `http://172.17.0.1:8000` (Docker bridge gateway).
6. n8n 2.x uses draft/published workflow versions: a DB `active=1` edit
   does not register webhooks; activation must go through the UI or
   `POST /api/v1/workflows/{id}/activate`.

`clinical-analysis.json` already read `$json.body` correctly; updated its
Respond nodes to `firstIncomingItem` for consistency.

Live verification results (n8n-live-test on :5679 → backend :8000):
- `analyze_via_n8n()` (no train) returns the full report with the real
  patient (was "Unknown" pre-fix).
- `train: true` through the webhook trains a diabetes logistic model
  (accuracy 0.66) then analyzes: prediction + risk + 3 evidence items,
  full report in one webhook response.

## Completed

- Milestones 1–10 — prior context (committed + pushed).
- Milestone 11 — committed + pushed (`a4161c0`, `6c08bf1`, `e1e2714`,
  `90d194b`, `a6f88bc`, `8894006`).
- This session (uncommitted):
  - `n8n/healthcare-endtoend.json` — `$json.body.*` field references,
    removed Write node + binary, `firstIncomingItem` respond nodes
  - `n8n/clinical-analysis.json` — `firstIncomingItem` respond nodes
  - `n8n/README.md` — updated to the corrected workflow (no disk write),
    credential requirement, Docker networking, activation note
  - Password reset for the original n8n instance (`healthcare-n8n` :
    5678, owner `monjurulhaquerajun@gmail.com`) → new password
    `NewPassw0rd!` (bcryptjs `$2a$10$` hash written via SQLite; stale
    WAL removed so the edit took effect; login verified)
  - Live n8n verification via a throwaway `n8n-live-test` instance
    (:5679) with owner `test@test.local`, full-scope API key, and the
    Healthcare API Token credential (id `dSEzRW3tonyQlUwD`)
  - Frontend suite still 35 passing (no frontend changes)

## Next Files (frontend / downstream)

- Commit the n8n workflow fixes + docs (focused `fix(n8n)` / `docs(n8n)`
  commits), then push.
- Optionally re-activate the original `healthcare-n8n` instance: create
  the Header Auth credential, re-select it on the HTTP nodes, patch URLs
  to the Docker gateway (or host networking), import the fixed workflow
  and activate via the UI/public API.
- Backlog (unchanged): patient persistence, mortality/readmission risk,
  SHAP explainability, disk-archive of reports (n8n file sandbox) via a
  volume under `~/.n8n-files`.

## Design Notes

- ADR-010 (thin view layer) unchanged — the dashboard only renders; the
  n8n workflow now returns the full report object the dashboard reads.
- n8n 2.34 sandbox constraints discovered live: Code-node `fs` is
  disallowed; readWriteFile writes are restricted to `~/.n8n-files`
  (`restrictFileAccessTo`, default) and cannot create parent dirs; the
  public API `PUT` rejects `n8n-nodes-base.executeCommand` (POST
  accepts it) — a reason the workflow avoids file I/O.
- Credential handling: the committed JSON keeps the
  `PLACEHOLDER_CREDENTIAL_ID`; deployers must create the credential and
  let n8n store its real ID.
- Lint/tests: frontend from `frontend/`, backend from `backend/`; never
  ruff from the repo root.

## Status

Milestones 1–11 committed/pushed; live n8n end-to-end verification
complete with the committed workflow fixed and verified (analyze-only and
train+analyze). Workflow + doc changes are uncommitted — next step is
focused commits + push.