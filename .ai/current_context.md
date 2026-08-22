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

---

## M3 Addendum (complete)

1. Anonymization wired at both ingestion choke points
   (`canonical.load_canonical_frame`, `hospitals.build_hospital_sites`);
   sepsis `insurance` column verified dropped pre-training.
2. `scripts/run_m3_evaluation.py`: §12 RAG + Agent metrics and §13
   baselines B2-B5 computed & stored (`artifacts/experiments/m3_*`).
   Faithfulness threshold shown miscalibrated for TF-IDF via raw-cosine
   diagnostics (template .064 vs ceiling .337); use dense embedder for
   RAGAS-style numbers.
3. Baselines recorded: B2 FL-only acc .9250 evidence 0; B3 FL+RAG
   evidence 3 completeness 1.00; B4 consistent=True; B5 full stack with
   n8n probe. All five proposal baselines now on record (B1 in M2 file).
4. `n8n/clinical-full.json` — single 10-step flowchart workflow, active
   in local n8n; verified: structured rejection, low-risk silent path,
   high-risk path fires DOCTOR_NOTIFY_WEBHOOK while still returning the
   stored clinical report (`notified: true`). Owner login configured on
   the local instance.
