# Next Session

## Objective

Fix the P0 correctness/security defects (see `.ai/NEXT_TASK.md` §1), then
re-run the baseline study to confirm nothing regressed. Do NOT re-derive
project history — read `.ai/NEXT_TASK.md` for the full prioritized list and
`.ai/current_context.md` for entry points.

## Done recently (2026-09-10)

1. Restored the stripped test suite + `scripts/baseline_study.py` from the
   pre-strip commit `6630867^`, adapted to current main. Fixed two real
   regressions the restored tests exposed: `metrics._output_of` over-counting
   empty-output dicts (task_completion_rate inflated), and
   `crew._parse_report` missing `@staticmethod`. Backend 309 pass + frontend
   58 pass + ruff clean. Commit `6b75585`.
2. Re-ran the baseline study; found + fixed a real kidney label-inversion bug
   (`split_dataset` dropped `preset=preset`, so string labels oriented
   alphabetically instead of disease-positive). Kidney 0.020 → 0.980 acc.
   Added a regression test + refreshed the Findings doc. Commit `4cef96e`.
3. Rewrote `.ai/NEXT_TASK.md` as a self-contained handoff.

## Next steps

1. `.ai/NEXT_TASK.md` §1 P0 items (mTLS/plaintext gRPC, hospital-site
   clobber, train-dataset path traversal, image z-score).
2. Re-run `DATASET_DIR=~/dataset ./CrewAI/.venv-opencode/bin/python scripts/baseline_study.py`
   after each change to catch label-orientation regressions.

## Do not start without asking

- Commit / push actions
- Dashboard redesign
- n8n workflow changes
- New agents / diseases
- Federated architecture changes (beyond the P0 cert/site-clobber fixes)
- Anything requiring a CrewAI/LLM API key — ask the user.