# Changelog

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
