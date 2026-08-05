"""Federal Learning + Privacy module (Phases 2 & 3).

Simulates a multi-hospital federated training fleet:

- Phase 2: PII anonymization, differential privacy (Opacus), secure aggregation,
  and privacy metrics (epsilon, membership-inference resistance, leakage rate).
- Phase 3: local models (PyTorch MLP / XGBoost) trained per hospital and
  aggregated via Federated Averaging (FedAvg) into a global model.

Raw patient data never leaves a hospital: only (masked, DP-protected) model
updates are exchanged.
"""

__all__ = [
    "data",
    "models",
    "privacy",
    "server",
    "train",
    "predict",
]