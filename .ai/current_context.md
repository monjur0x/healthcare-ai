# Current Context

## Current Milestone
M2 — Federated Learning made experimentally real (heterogeneous multi-disease federation per proposal).

## Current Task (complete)
Fixed the critical data-overwrite bug + implemented canonical-schema federation:

1. BUG: `build_hospital_sites` overwrote all four hospitals' specialty CSVs
   with partitions of one preset whenever `python -m federated run` executed.
2. FIX: new `--heterogeneous` mode (`federated/__main__.py`, `distributed.py`)
   uses each hospital's own `data/hospitals/<id>/data.csv` as-is — never
   partitions or writes. Verified byte-identical (md5) across runs.
3. NEW `federated/canonical.py`: shared 11-feature schema from the proposal's
   "Expected Inputs" (age, gender, bmi, blood_pressure, heart_rate, spo2,
   glucose, creatinine, cholesterol, hemoglobin, albumin) + binary
   `has_disease` target; per-disease adapters map diabetes/heart/CKD/sepsis
   columns onto it so FedAvg weight shapes always match.
4. `scripts/run_m2_experiment.py`: stratified pooled split; centralized vs
   Flower-federated comparison; stores metrics to
   artifacts/experiments/m2_results.{json,md} + both model artifacts.
5. Distributed global model now persists `feature_names`.

## Results (stored)
Centralized: acc .9119, ROC-AUC .9455, F1 .7514, MCC .7109
Federated 10r: acc .8894, ROC-AUC .9104, F1 .7128, MCC .6462
Flower gRPC 5r (multi_disease): client-weighted acc .8840, log loss .403→.361
Federated convergence: acc .742 → .889 across rounds; ~1.8 MB exchanged.

## Status
M2 Definition of Done satisfied. Lint clean. Next backlog: data
anonymization stage, shared-scaler study, n8n multi-step workflow,
dashboard risk panel wiring.
