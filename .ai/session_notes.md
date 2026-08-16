# Session Notes — 2026-08-15

## Task

Rework the Streamlit dashboard into a doctor-facing Clinical Decision
Support UI (Milestone 11). No backend changes allowed; n8n stays core;
honest handling of unsupported outputs; update docs at the end.

## What happened

1. Read AGENTS.md-required docs (SYSTEM_SPECIFICATION/API_REFERENCE are
   empty) and explored the backend `ClinicalReport` shape, the n8n
   workflows, and the old dashboard. Established baselines: backend 326
   passing, frontend 13 passing.
2. Added `frontend/dashboard/clinical.py` (pure helpers, 14 tests).
3. Extended `frontend/dashboard/client.py` with `analyze_via_n8n`,
   `n8n_health`, shared `_analyze_payload` (6 new tests).
4. Rewrote `frontend/streamlit_app.py` as a 5-tab CDS dashboard.
5. Updated `n8n/healthcare-endtoend.json` Code node to return the full
   `report` in the webhook response (JSON validated).
6. Fixed frontend test failures:
   - `ElementList` is not importable from `streamlit.testing.v1` —
     simplified `_texts()` to iterate element lists directly.
   - `is_flag_feature("mechanical_ventilation")` was False — added it to
     `FLAG_FEATURES`.
   - `StreamlitDuplicateElementId` on the report download button (the
     report renders on both Assessment and Results tabs) — per-call-site
     `download_key`.
   - `file_uploader.set_value` needs a `(name, bytes, mime)` tuple and a
     real image — generated a PIL PNG in the test.
   - The smoke test asserted a message ("no prediction model was
     configured") that only appears when risk is absent; asserted the
     actual graceful form message instead.
   - Black does not split long lambdas — replaced inline lambdas with
     local `fake_analyze` functions.
7. Tooling: added `frontend/pyproject.toml` mirroring the backend config.
   Root cause: ruff resolves config from cwd while black/isort resolve
   from the file's location, so the frontend previously linted with
   defaults and disagreed with the backend config. Now frontend lints
   standalone from `frontend/`.
8. Final checks: frontend 35 passing, backend 326 passing, ruff/black/
   isort clean on both trees.
9. Live validation (backend started with the run-script env vars):
   - `/api/v1/model` → brain-tumor CNN only (`feature_names: null`), so
     the assessment tab shows the graceful "no tabular model" state.
   - `client.analyze` (no tabular model) → report with 3 evidence items,
     no prediction/risk (graceful path works).
   - `client.analyze_image` (real glioma scan) → prediction
     `meningioma` @ 69% + `high` risk + monitoring schedule + evidence.
   - n8n not running locally → n8n webhook path verified only via
     hermetic unit tests + workflow JSON validation (recorded in docs).

## Decisions / notes

- Report is rendered by `render_clinical_results` on the Assessment and
  Results tabs (and Imaging) in the same run — unique widget keys are
  required. Verified the duplicate-ID failure mode.
- `frontend/pyproject.toml` is a deliberate addition so the frontend has
  its own tooling config (same rules as backend). Do NOT run ruff from
  the repo root; frontend lint from `frontend/`, backend lint from
  `backend/`.
- Patient persistence, mortality/readmission risk, and SHAP-style
  explainability are deferred (recorded in BACKLOG); the dashboard shows
  honest "not estimated / future work" states.

## Open items

- Commit Milestone 11 (working tree is dirty) — user has not requested a
  commit; ask before committing.
- Live n8n end-to-end verification from the dashboard needs a running
  n8n instance.
## Session: live n8n verification + original-instance password reset

- Reset the forgotten owner password on `healthcare-n8n` (:5678):
  generated a bcryptjs `$2a$10$` hash, wrote it into a copy of the
  container's `database.sqlite` (SQLite), then docker cp'd it back and
  removed the stale `-wal`/`-shm` (the container's old WAL was
  overriding the edited main file). New password: `NewPassw0rd!`
  (owner `monjurulhaquerajun@gmail.com`); login verified via
  `POST /rest/login`.
- Live n8n E2E (n8n 2.34.6, throwaway `n8n-live-test` on :5679):
  - Public API auth: `N8N_API_KEY` env is ignored; API keys must be
    created by an owner via `POST /rest/api-keys` (private API) and sent
    as `X-N8N-API-KEY`. Owner setup is `POST /rest/owner/setup`; login
    field is `emailOrLdapLoginId`.
  - CLI `import:workflow` deactivates workflows and needs a uuid `id`;
    DB `active=1` alone does NOT register webhooks (draft/published
    model). Activate via UI or `POST /api/v1/workflows/{id}/activate`.
  - Webhook node nests the payload under `body` — the committed workflow
    read `$json.*`/`item.json.*` at top level, so `train`, `patient`,
    `features` were undefined (train never ran; patient "Unknown").
    Fixed to `$json.body.*` / `item.json.body.*`.
  - readWriteFile write fails on missing dirs (no folder creation) and
    is restricted to `~/.n8n-files`; Code-node `fs` is disallowed; the
    public API `PUT` rejects `n8n-nodes-base.executeCommand`. → Removed
    the Write-to-disk + binary from the workflow; the respond node now
    returns the full report (dashboard contract).
  - `respondWith: allIncomingItems` returns an array; the dashboard's
    `analyze_via_n8n()` expects a dict → both respond nodes now use
    `firstIncomingItem`.
  - Docker networking: HTTP nodes must target `http://172.17.0.1:8000`
    (bridge gateway) from inside the n8n container, not `localhost:8000`.
  - Verified live: `analyze_via_n8n()` (report + real patient) and
    `train: true` (diabetes logistic 0.66 → prediction + risk + evidence
    in one response).

## Original-instance activation (same session, :5678)

- Reset password `NewPassw0rd!` works for the owner login; owner id
  `3614f42f-5fa4-4a14-8c9e-3dfc9c6317f4`.
- n8n 2.34 API-key creation (`POST /rest/api-keys`) requires BOTH
  `expiresAt` (unix seconds) and the exact valid scope set — a superset
  with `workflow:execute` / `variable:*` / `execution:delete` /
  `user:read` fails with "Invalid scopes for user role". Valid set used:
  credential:create/read/list/update/delete, workflow:create/read/
  update/activate/deactivate/list, user:list, execution:read/list.
- The create response masks `apiKey` (e.g. `******Ebk0`) and returns the
  raw JWT in `rawApiKey` — use that as `X-N8N-API-KEY`.
- Credential `Healthcare API Token` created via public API: id
  `6bjqNVT4MoPaTZ6L` (httpHeaderAuth, Authorization: Bearer
  healthcare-ai-dev-token).
- Workflow id `e2f0a94c-90ee-4f9f-9b39-4d6bfd71b4e2` updated via
  `PUT /api/v1/workflows/{id}` with the repo's fixed JSON (URLs patched
  to `http://172.17.0.1:8000`, credential wired) then activated via
  `POST .../activate` (public API with the key). GET on the POST-only
  webhook returns 404 (expected).
- Verified live on :5678: analyze-only (Nora Kim, 3 evidence),
  train+analyze (logistic, accuracy 0.683), missing-features error →
  proper single-object JSON; dashboard `analyze_via_n8n()` against
  :5678 returns the real report.
- Backup copies: key raw JWT at /tmp/opencode/n8n_apikey_orig.txt;
  deployment payload /tmp/opencode/wf_orig_deploy.json.

## Blood-pressure SYS/DIA input (frontend)

- Clinical Assessment's Blood Pressure field was a plain number input.
  The user wanted "120/90" (systolic/diastolic) accepted.
- The model's `bloodpressure` feature is the PIMA diabetes "Blood
  Pressure (mm Hg)" column = **diastolic**, so `SYS/DIA` maps to the
  diastolic component (`120/90` → `90`); a lone number is used as-is.
  Recorded in the widget help text + parser docstring.
- Added `parse_blood_pressure(raw) -> float | None` to
  `frontend/dashboard/clinical.py` (pure, exported); rejects empty,
  non-numeric, >2 parts, `dia > sys`, and non-positive values.
- `streamlit_app.py`: `feature_widget` special-cases `bloodpressure`
  via `_blood_pressure_widget` (text_input default "120/80", help text,
  st.error + fallback 80.0 on invalid). Same widget keying as before so
  session state persists.
- Tests: +3 in `test_clinical.py`; frontend suite 38 passing; lint clean.
- If the project ever needs systolic as a separate marker, the risk
  service's marker thresholds would need a `systolic_bp` key — not
  implemented (features dict must stay exactly the model's columns).
