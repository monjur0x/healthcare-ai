# Next Task

## Immediate follow-ups (from disease-context fix, 2026-08-26)

1. **LLM-path spot check** — run one full `crew.run()` with the
   OpenRouter key active on a high-risk diabetes input; confirm the
   merged narrative stays diabetes-specific (the deterministic base is
   verified; `_merge_llm_over_base` could still pull off-topic content
   from the model's own text).
2. **Heart/kidney/sepsis spot check** — same verification pattern
   (3 inputs each) when those presets are served; the registry and
   playbooks are already in place but only diabetes was exercised.
3. **`API_ACTIVE_PRESET` in `.env.example`** — document the new env var
   alongside `API_MODEL_PATH`.
4. **Commit** — all changes from this fix are uncommitted at user
   request.

## Carried-over deferred items

From the perf(crewai) session (see `.ai/next_session.md`):
- RAG query-building was FIXED as part of the disease-context task
  (was: class-int query) — item #1 there is resolved.
- Risk-scoring nuance remains: score = positive-class probability by
  design; marker thresholds feed `risk_factors`, not the score.
- LLM narrative faithfulness: partially mitigated — agents now receive
  `disease_context` in kickoff inputs, but task descriptions still
  embed only demographics. Re-test after LLM-path spot check.
- Free-tier ε re-measurement for the privacy CHANGELOG entry.

## Do not start without asking
- Dashboard redesign
- n8n workflow changes
- New agents / diseases
- Federated architecture changes
