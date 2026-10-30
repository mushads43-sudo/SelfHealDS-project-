# SelfHealDS — Explainable Data Quality Intelligence

SelfHealDS is a local-first, explainable preprocessing and self-healing framework for structured tabular data. Its research contribution is the controlled decision protocol that separates detection, evidence, repair authorization, post-repair validation and quarantine. It is designed for the quality-control stage that normally occurs before analytics, reporting, statistical analysis or machine learning.

## Core workflow

Upload → Profile → Detect → Explain → Decide → Repair → Revalidate → Quarantine → Audit → Export

The safety principle is simple: **the system must not change a value merely to make the dataset look clean.** Repairs require deterministic evidence; ambiguous findings remain visible and can be quarantined.

## Detection coverage

- Missing values and textual placeholders
- Duplicate records and duplicate normalized column names
- Constant fields
- Negative/impossible ages
- Negative, malformed and incorrectly formatted phone identifiers
- Email structure
- Numeric parsing and type consistency
- Negative quantities
- Date validity and canonicalization
- City/country semantic consistency and reference-vocabulary anomalies
- Mixed measurement units and deterministic unit conversion
- Whitespace normalization
- IQR-based statistical outliers

## Research-oriented capabilities

- Explicit quality taxonomy and rule identifiers
- Evidence and human-readable reasoning for each finding
- Deterministic repair policy with confidence/evidence score
- Post-repair validation gate
- Review-required quarantine output
- Cell-level repair provenance and audit trail
- Dataset fingerprinting for reproducibility
- Data-quality passport in JSON
- Controlled corruption benchmark with ground truth
- Precision, recall, F1, repair accuracy, unsafe-repair rate and clean-input false-positive rate
- Corruption-level sensitivity sweep
- Minimal baseline comparison
- Run timestamp, evaluation version and random seed attached to research metrics

## Scientific limitations

The framework does not claim to detect every possible data-quality problem. Semantic checks depend on declared reference vocabularies and column semantics. Statistical outliers are signals of unusualness, not proof of incorrectness. Synthetic corruption benchmarks are controlled experiments, not substitutes for independently curated production data.

## Run locally

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

No paid AI API is required for the core system.

## Publication protocol

Use the Research Evaluation page and `scripts_run_benchmark.py` to generate run-specific evidence. Never copy illustrative or historical percentages into a paper without recording the dataset fingerprint, corruption protocol, seed, evaluation version, metric definition and UTC timestamp.


## Hybrid anomaly detection
SelfHealDS uses Isolation Forest (200 trees, deterministic seed 42) as a statistical anomaly flagger for sufficiently sized numeric datasets. It does not perform repairs; deterministic rules and the safety gate remain responsible for repair decisions.


## Repair reasoning policy

SelfHealDS does not treat every detected violation as permission to invent a value. Each repair must have an explicit derivation.

### Example: missing age

Suppose an `age` column contains the observed values:

`21, 24, 28, 31, 47, missing`

There are five usable observations, so the minimum-observation policy is satisfied. Missingness is `1/6 = 16.67%`, which is below the configured 20% automatic-imputation ceiling.

The sorted observed ages are `21, 24, 28, 31, 47`; therefore the median is `28`. The system replaces the missing age with **28**, not because 28 is known to be the person's true age, but because:

1. age is a numeric field with a non-negative domain;
2. the fallback is derived entirely from observed values;
3. the median is less sensitive to an extreme observation than the mean;
4. the column has enough observed values under the declared policy;
5. missingness is below the declared policy ceiling;
6. the exact calculation, evidence and policy are recorded in the audit trail.

The replacement remains an estimate, not recovered ground truth.

If the column has too few usable observations or exceeds the missingness ceiling, SelfHealDS refuses automatic imputation and sends the case for review.

### Example: Mumbai–France

If `city = Mumbai` and `country = France`, the system uses the declared reference relationship `Mumbai -> India`. Because the relationship is an exact reference lookup, it has stronger evidence than a statistical guess. The country may therefore be changed to `India`, with the mapping and original value recorded in the audit event.

### Example: invalid phone `123`

The system detects that `123` contains only three digits. The declared generic structural range is 7–15 digits. Missing digits cannot be inferred safely, so the value is **not fabricated** and the record is eligible for quarantine/review.

### Example: statistically unusual value

Isolation Forest may identify an unusual numeric observation. That is treated as statistical evidence only. It is not automatically changed because unusualness does not establish the correct replacement.

## Real-World Case Studies

The UI separates three dataset-entry areas:

1. **Sample Datasets** — the six controlled project demonstrations.
2. **Real-World Case Studies** — curated public datasets/cases with provenance, documented issue descriptions, natural-vs-controlled distinction, before/after evidence, and case-specific validation rules.
3. **Upload Your Own Dataset** — the general CSV workflow.

The current real-world catalog covers aerospace (NASA Mars Climate Orbiter as a documented incident; exact historical telemetry is not falsely claimed), healthcare, transportation, government/food safety, environmental sensing, census/government data, and business transaction semantics. Public files are downloaded on demand and cached locally; the source URL, dataset fingerprint and run timestamp should be recorded for paper experiments.

For cases with independently curated ground truth, the UI reports recomputed precision/recall/F1 and repair accuracy. For cases without cell-level ground truth, SelfHealDS deliberately reports natural issue and before/after quality evidence without inventing precision/recall.

## Research-grade closed-loop features
- Evidence-backed repairs with rule IDs and evidence-strength scores.
- Root-cause grouping for related findings.
- Post-repair cell validation and repair-status tracking.
- Explicit do-no-harm review boundary for uncertain cases.
- Repair provenance suitable for audit and reproducible research.
- Research Evaluation with labelled corruption, baseline comparison, ablation and sensitivity sweeps.

### Research claim discipline
SelfHealDS is not claimed to make every individual cleaning operation uniquely novel. Novelty should be established through the paper's literature review. The project's differentiator is the integrated evidence-driven, guarded, validated and auditable self-healing workflow and its measurable evaluation protocol.
