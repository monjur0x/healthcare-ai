# Privacy Notes — Known Limitations and Honest Scope

This document describes what the privacy layer **actually does**, what it
**does not do**, and where the current implementation falls short of the
research proposal's aspirational goals. It is maintained alongside the code
so reviewers can verify that no claim overstates the implementation.

---

## 1. Differential Privacy (Opacus DP-SGD)

### What it provides

Local training uses Opacus `PrivacyEngine.make_private()` with per-sample
gradient clipping (`max_grad_norm`) and Gaussian noise injection
(`noise_multiplier`). The accountant reports a per-call epsilon via
`PrivacyEngine.get_epsilon(delta)`.

### What it does NOT provide

- **Torch-only**: DP-SGD is only supported for models exposing a torch
  `nn.Module` (currently `TorchMLPClassifier`). The scikit-learn MLP path
  (`TabularClassifier`) does NOT apply DP.
- **Per-round, per-client accounting**: each client creates a fresh
  `PrivacyEngine` for every federated round. The Opacus accountant state is
  lost between rounds because the model is re-created from the aggregated
  global weights. As a result, the reported cumulative ε is a **naive sum**
  of per-round values, which is a conservative upper bound (not tight RDP
  composition).
- **No secure DP aggregation**: DP-SGD is applied locally; noise is added
  before weight transmission. There is no distributed DP noise sharing.

### Epsilon composition method

The experiment script labels every cumulative ε with its composition method:

| Method | Description | When used |
|--------|-------------|-----------|
| `"single_round"` | Only one round was executed; the reported ε is directly from the accountant. | rounds = 1 |
| `"naive_sum_upper_bound"` | Sum of all per-round epsilons. Valid upper bound under basic composition; may overestimate by √k or more vs RDP. | rounds > 1 |
| `"rdp_accountant"` | Tight Rényi-DP composition from a persistent accountant state. | NOT currently implemented |

The JSON output always includes `epsilon_composition_method` so downstream
consumers can distinguish upper bounds from tight compositions.

---

## 2. Secure Aggregation

### What it provides

A pairwise one-time-pad masking scheme: each pair of clients derives an
identical random mask from a shared seed; client *i* adds mask(i,j) while
client *j* subtracts it. When summed on the server, all masks cancel and the
server obtains the exact mean without seeing any individual update.

This is implemented in `federated/privacy.py::SecureAggregator`.

### What it does NOT provide

- **Not Bonawitz-style SecAgg**: there is no secret-sharing, no threshold
  cryptography, no dropout tolerance. If any client drops mid-round, the
  masks cannot be reconstructed and the round must be restarted.
- **Not production cryptographic security**: the pairwise seeds are derived
  from deterministic indices (`100_000 + lo * 7919 + hi`), not from a key
  exchange protocol. An adversary who observes two clients' indices can
  derive their shared mask.
- **Equal weights required**: the OTP masks cancel only under uniform
  aggregation weights. Weighted FedAvg is incompatible with this scheme.
- **Server sees the sum, not individual updates** — but a honest-but-curious
  server can potentially infer information from multiple rounds of sums.

### Correct claim

> "Secure Aggregation uses pairwise one-time-pad masking to prevent the
> aggregation server from observing any single hospital's raw model update,
> provided all clients complete the round and use equal weights."

### Incorrect claims (do NOT make)

- ~~"Production-grade Secure Aggregation"~~
- ~~"Bonawitz-style secure aggregation"~~
- ~~"Dropout-tolerant secure aggregation"~~
- ~~"Cryptographically secure against active adversaries"~~

---

## 3. Membership Inference Attack (MIA)

### What it provides

A confidence-based train-vs-holdout attack using the model's predicted
probability as the membership signal. AUROC is computed via the rank-sum
(Mann–Whitney U) statistic.

### What it does NOT provide

- This is a **simplified baseline MIA evaluation**, not a comprehensive
  attack suite. It does NOT include:
  - Shadow-model attacks (Shokri et al., 2017)
  - Label-only attacks (Choquette-Choo et al., 2021)
  - Gradient-based attacks (gradient inversion)
  - Per-class MIA analysis
  - Calibration-based attacks (entropy, modified entropy)
- The attack uses only the max softmax probability; stronger signals
  (logit magnitude, loss value) are ignored.
- Results may underestimate vulnerability compared to a real adversary with
  auxiliary knowledge.

### Sample-size limitation

When train-member or holdout-nonmember counts fall below ~50 samples, the
AUROC estimate has wide confidence intervals and should be interpreted with
caution. The experiment script records sample counts in
`mia_sample_counts` so reviewers can assess statistical significance.

---

## 4. Data Leakage Rate

### What it provides

`inspect_federation_payloads()` inspects the actual numpy arrays transmitted
between clients and server for:
1. Non-float dtypes (which could encode string data)
2. NaN / Inf values (side-channel encoding)
3. Suspiciously large magnitudes (> 1e6, possibly encoded identifiers)

It returns a measured leakage rate based on how many payloads pass these
checks, along with per-payload inspection evidence.

### What it does NOT provide

- It cannot detect leakage embedded in the *statistical* properties of
  model weights (e.g. gradient inversion attacks).
- It cannot verify that feature names were stripped from payload metadata.
- It does not audit the communication channel itself (TLS termination,
  log files, memory dumps).

### Current measured result

In the experiments run on this repository's bundled datasets, the measured
leakage rate is **0.0000** because:
1. Only float32/float64 model parameter arrays are transmitted.
2. No NaN, Inf, or extreme-magnitude values were detected.
3. Raw patient rows never leave the hospital process.

This is a genuine measurement, not an assumption. The evidence is recorded
in the output JSON under `measured_privacy_metrics.payload_inspection`.

---

## 5. Anonymization Coverage

`anonymize_frame()` strips columns whose names contain PII-pattern substrings
(name, patient, dob, ssn, mrn, insurance, etc.). It runs at these locations:

| Data path | Anonymization applied? | Location |
|-----------|----------------------|----------|
| Canonical schema loading (heterogeneous) | ✅ Yes | `federated/canonical.py::load_canonical_frame` |
| Single-preset partitioning | ✅ Yes | `federated/hospitals.py::build_hospital_sites` |

Verified: the sepsis dataset's `insurance` column is stripped before any
local training occurs (tested with hospital_D data).

### Limitation

Pattern-matching column names catches common PII fields but cannot detect
PII embedded in free-text cells or encoded within numeric values.

---

## 6. Simulated Hospital Partitioning

Hospitals A–D are simulated by partitioning public datasets (Pima Diabetes,
UCI Heart, UCI CKD, synthetic sepsis). They are NOT real hospitals with real
patient data. Privacy guarantees demonstrated here apply to the federated
learning mechanism, not to real-world clinical deployment.

Key differences from production:
- All processes run on the same machine (localhost gRPC)
- Datasets are public benchmarks, not PHI
- No network-level adversaries
- No regulatory compliance requirements (HIPAA/GDPR) tested

---

## 7. Secure Mode

Opacus is initialized with `secure_mode=False`. Setting it to `True`
requires cryptographically secure randomness (RNG) and disables some
optimizations. For research purposes, `secure_mode=False` is sufficient;
for production deployment, `secure_mode=True` should be used and validated.

---

## 8. Summary Table

| Mechanism | Implemented | Production-grade | Key limitation |
|-----------|------------|-----------------|----------------|
| DP-SGD (Opacus) | ✅ | ⚠️ Research | Torch-only; ε composition is upper bound |
| Secure Aggregation | ✅ | ❌ Prototype | Pairwise OTP, no dropout tolerance |
| Data Anonymization | ✅ | ⚠️ Pattern-match | Column-name patterns only |
| MIA Audit | ✅ Baseline | ⚠️ Simplified | Confidence-based only, no shadow models |
| Data Leakage Audit | ✅ Measured | ⚠️ Structural checks | Cannot detect gradient inversion |
| Encrypted Transport (TLS) | ✅ Configurable | ⚠️ Not default | Requires cert generation |
