# Changelog

## perf(crewai): reduce free-tier LLM execution time

**2026-08-25**

### Changes
- `LLM_MAX_ITERATIONS` default 10 → 4 (each iteration is a full LLM
  round-trip; not a correctness requirement).
- `evidence_retrieval` and `explanation` tasks now run with
  `async_execution=True` — they share no data dependency. Verified in
  crewai 1.15.17 source (`crew.py::_execute_core`): sync tasks wait for
  all pending asyncs before executing, so `treatment`/`report_generation`
  still receive their outputs.
- New `LLM_MAX_TOKENS` (default 1024) wired into both LLM branches
  (custom-endpoint dict + `Agent(max_tokens=...)`).
- New `LLM_TIMEOUT_SECONDS` (default 120) wired as endpoint `timeout`
  and `Agent(max_execution_time=...)`. Defaults calibrated against the
  free-tier reality: 40 s killed queued calls (silent fallback to the
  deterministic report); <1k tokens truncated tool-call JSON into empty
  completions.

### Measured before/after
- **Baseline** (pre-change, `stealth/ox-alpha` via OpenRouter free pool):
  start 02:12:15 → end 02:24:53 = **12 m 38 s**, full 7-agent LLM path.
- **Tuned re-measure: blocked by upstream throttling tonight.** Direct
  probe of the endpoint returned HTTP 429 ("temporarily rate-limited
  upstream, shared pool") on 2 of 3 trivial requests; four full runs
  (02:49, 02:54, 03:01, 03:11) each aborted partway when litellm
  surfaced the 429s as empty completions, falling back to the
  deterministic report (~4–7 min of attempts before fallback).
- Structural expectation from the changes alone: 5 sequential blocking
  stages instead of ~7 agent rounds, with per-agent iterations capped at
  4 — a successful run should land around half the baseline. To be
  re-measured when the shared pool frees up (noted in
  `.ai/next_session.md`).

### Config verified without provider
```
max_iter: 4 | max_tokens: 1024 | timeout: 120
custom llm dict keys: [api_key, base_url, custom_openai,
                       max_tokens, model, temperature, timeout]
agent max_iter=4 max_tokens=1024 max_execution_time=120
async flags: evidence_retrieval=True, explanation=True, rest=False
```

## feat(privacy): research-complete, reproducible privacy layer (§8 metrics)

**2026-08-25**

### Measured Headline Numbers

| Metric | DP + SecAgg | SecAgg only | Method |
|--------|------------|-------------|--------|
| Cumulative ε (5 rounds) | 45.3607 | N/A | naive_sum_upper_bound |
| Per-round ε range | 0.481–3.585 | N/A | Opacus accountant |
| MIA AUROC | 0.5011 | 0.5004 | confidence-based baseline |
| Attack Resistance Score | 0.9978 | 0.9992 | 1 − 2(AUROC − 0.5) |
| Data Leakage Rate | 0.0000 | 0.0000 | payload inspection |

### Added
- `federated/privacy.py`: `compute_cumulative_epsilon_upper_bound()`,
  `inspect_federation_payloads()`; enhanced `privacy_metrics_summary()`
  with per-round ε tracking, composition method label, MIA sample counts,
  and payload inspection evidence.
- `federated/server.py`: `FedAvgServer` now tracks per-round epsilons,
  exposes `cumulative_epsilon_upper_bound` and `payload_inspection`
  properties, runs `inspect_federation_payloads()` after final round.
- `api/services.py`: `_train_federated` measures data leakage from actual
  federation payloads via `inspect_federation_payloads` instead of
  hardcoding `[{"exposed": False}]`.
- `scripts/run_privacy_experiment.py`: reproducible privacy experiment
  runner that persists seed, config, and all metrics as JSON.
- `docs/PRIVACY_NOTES.md`: honest documentation of what the privacy layer
  does and does not protect against.
