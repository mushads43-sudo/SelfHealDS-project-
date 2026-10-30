# SelfHealDS Architecture

```text
                           Streamlit Research UI
                                    |
                          backend.pipeline.assess
                                    |
             +----------------------+----------------------+
             |                      |                      |
          profiler              detector                rules
             |                      |                      |
       schema/profile       multi-dimensional      semantic maps /
       fingerprint          explainable rules       deterministic parsers
             |                      |                      |
             +----------------------+----------------------+
                                    |
                             decision layer
                                    |
                         repair authorization
                                    |
                           backend.repair
                                    |
                          post-repair detector
                                    |
                 +------------------+------------------+
                 |                                     |
          validated clean                       unresolved errors
                 |                                     |
              export                               quarantine
                 |                                     |
                 +------------------+------------------+
                                    |
                              audit / passport

Research layer:
input -> controlled corruption -> ground truth -> detector -> scoring
                                       |
                            baseline + corruption sweep
```

## Design principles

1. **Detection is separate from repair.** A finding does not automatically authorize a modification.
2. **Evidence precedes automation.** High-confidence deterministic transformations may be applied; ambiguous cases are preserved for review.
3. **Validation is authoritative.** The clean export is produced only after a second detection pass.
4. **Provenance is first-class.** Repair logs contain row, column, original value, repaired value, reason, confidence, rule ID and status.
5. **Reproducibility is explicit.** Dataset fingerprints, UTC timestamps, evaluation version and benchmark seed accompany research results.
6. **Claims are bounded.** The finite rule catalogue and reference vocabularies are visible to the user and documented for the research paper.


## Repair-decision hierarchy

The system follows a conservative evidence hierarchy:

1. **Representation-preserving transformations** — trimming whitespace, canonicalizing case or separators.
2. **Deterministic transformations** — unit conversion, exact alias normalization, unambiguous date parsing.
3. **Constraint-preserving sign correction** — only where the field domain makes the sign invalid and the magnitude remains within the declared domain.
4. **Observed-data imputation** — only when the missingness and minimum-observation policies are satisfied; the exact fallback calculation is recorded.
5. **Reference-based semantic correction** — only when an exact declared cross-field relationship supplies the replacement.
6. **Statistical anomaly flagging** — review-only; no automatic replacement.
7. **Insufficient evidence** — quarantine/review.

The hierarchy prevents the system from using a weak statistical signal to justify a high-impact semantic modification.

## Evidence record

Each repair event contains:

- original row and column;
- original value;
- repaired value;
- issue type;
- rule ID;
- evidence-based reason;
- confidence/evidence score;
- confidence interpretation;
- action status.

This makes every modification auditable and suitable for qualitative error analysis in the research paper.

## Real-world evaluation layer

```text
Case-study catalog + provenance
             |
      public dataset loader
             |
      documented natural issues
             |
      case-specific constraints
             |
      existing detector/rulebook
             |
        repair + safety gate
             |
      post-repair validation
             |
   before/after + audit evidence
             |
 independent ground truth (when available)
```

Case-specific rules are activated by explicit context rather than hidden filename checks. This preserves the general upload workflow while allowing domain constraints such as provider-level consistency, documented missing sentinels and legitimate transaction semantics. Controlled corruption remains in the Research Evaluation layer and is never counted as a historical incident.
