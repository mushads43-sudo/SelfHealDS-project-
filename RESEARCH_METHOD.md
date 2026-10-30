# SelfHealDS Research Protocol

## Research question
Can a safety-aware, explainable data-quality framework detect heterogeneous
tabular corruption and improve data quality through deterministic, evidence-backed
repairs, while measurably limiting unsafe automatic changes — and can that
protocol be proven with the same rigor a reviewer would demand of any other
empirical claim?

## Claimed contribution
SelfHealDS combines multi-dimensional detection (structural, missingness,
type, format, domain, semantic, unit, statistical), evidence-based repair
authorization, post-repair re-validation, quarantine of unresolved errors, and
full repair provenance in one reproducible workflow. The contribution is the
**integrated decision protocol and its evaluation harness**, not a claim that
any individual technique (regex validation, unit conversion, Isolation
Forest, median imputation) is novel in isolation. See `POSITIONING.md` for the
full differentiation argument against both legacy cleaning tools and generic
LLM-based cleaning.

## Quality taxonomy
Findings are grouped into ten dimensions: Structural integrity, Missingness,
Uniqueness, Type consistency, Format validity, Format consistency, Domain
validity, Semantic consistency, Unit consistency, Statistical anomaly.

## Decision policy
Every finding carries: row/column location, observed value, severity,
confidence (evidence strength, not a calibrated probability),
human-readable rationale, proposed repair, and whether automatic repair is
authorized. Repairs are deterministic where possible; ambiguous cases are
never fabricated — unresolved Error-level records are quarantined into a
separate review dataset rather than silently dropped or guessed.

## Validation protocol
1. Profile the raw dataset.
2. Detect issues (rule engine + Isolation Forest statistical layer).
3. Apply only policy-authorized repairs.
4. Re-run the detector on the repaired data (closed-loop validation).
5. Remove unresolved Error-level records from the validated clean output.
6. Preserve those records in a review-required output with the reason attached.
7. Record every automatic change in an audit trail (rule ID, evidence, before/after value, validation status).

## Evaluation design
- Controlled corruptions are injected into the uploaded reference with
  known row/column/issue-type/expected-repair ground truth, and are never
  applied on top of a SelfHealDS-precleaned copy (avoids evaluation leakage).
- Detection precision, recall, F1 are computed against that ground truth.
- Clean-input false-positive rate is measured on cells that were never touched.
- Repair accuracy, safe-repair coverage and unsafe-repair rate are measured
  only over cases with a well-defined safe target; a non-inferable case
  (e.g. a 3-digit phone number missing unknown digits) can never be scored
  "correct" even if a value is emitted.
- The corruption mix includes both "obvious" (rule-violating) and "plausible"
  (structurally valid but numerically wrong) cases, to test the system's
  abstention boundary — recall on plausible cases is expected, by design, to
  be much lower than on obvious cases, since no rule can safely detect a
  subtly-wrong-but-valid value without ground truth.
- Results are compared against a declared minimal baseline (missing-value +
  exact-duplicate detection only) and against an ablation (deterministic
  rules alone vs. hybrid rules+ML).
- Every trial is repeated across 4 corruption levels (5/10/20/30%) x 5 fixed
  seeds (42-46), and every reported number carries a dataset fingerprint,
  evaluation-protocol version, seed and UTC timestamp.

## Metric definitions
- **Precision** = TP/(TP+FP): among newly predicted defects, what proportion match a known injected defect.
- **Recall** = TP/(TP+FN): among known injected defects, what proportion were detected.
- **F1**: harmonic mean of precision and recall.
- **Repair accuracy**: correct automatic repairs / automatic repair attempts.
- **Safe repair coverage**: correct automatic repairs / benchmark cases with a known safe target (a stricter, non-gameable denominator than repair accuracy alone).
- **Clean false-positive rate**: false positives / cells that were never deliberately corrupted.
- **Unsafe repair rate**: incorrect/unsafe automatic repairs / automatic repair attempts.

## Confidence and threshold semantics
Displayed "confidence" is an evidence-strength score assigned by the rule
policy - it is not a calibrated probability and must never be reported as
"98% certain" without a separate calibration study. Every numeric policy
threshold (age upper bound 120, phone digit range 7-15, missingness ceiling
20%, minimum 5 observations for a fallback, 1.5xIQR outlier fence, quality
score severity weights) is an explicit, documented engineering decision
exposed in the in-app Rule Catalogue - not an external statistical standard.

## Hybrid ML component
An Isolation Forest (200 trees, `random_state=42`) flags numeric statistical
unusualness only; it never authorizes or performs a repair, and its decision
score is not interpreted as a probability. This separation is what keeps the
system explainable: a statistical outlier is evidence to review, not proof of
an error.

## Research questions for a paper
RQ1. How accurately does SelfHealDS detect controlled heterogeneous corruptions?
RQ2. How does detection/repair behavior change as corruption severity increases?
RQ3. How often do automatically authorized repairs recover the known target without leaving an Error-level violation?
RQ4. What does the deterministic rule layer contribute versus the hybrid (rules + ML) system?
RQ5. Which error categories remain hardest, and why (see "plausible" corruption results)?

## Recommended paper result structure
1. Dataset and corruption protocol. 2. Baseline vs. SelfHealDS. 3. Detection
metrics by error category. 4. Corruption-level sensitivity (mean +/- sd across
seeds). 5. Repair accuracy and unsafe-repair rate. 6. Ablation study.
7. Error analysis with concrete false-positive/false-negative examples.
8. Runtime/scalability. 9. Threats to validity (see below).

## Threats to validity (be upfront about these in the paper)
- Synthetic corruption injection does not perfectly reproduce production
  error distributions; treat it as a controlled stress test, not a claim of
  real-world error prevalence.
- Reference vocabularies (countries, city<->country mappings, unit tables) are
  finite; a value outside them is "unrecognized," not proven wrong.
- The 7 real-world-adjacent case studies are documented public benchmarks
  (HoloClean Hospital/Flights, UCI Air Quality/Online Retail/Adult Census,
  Chicago Food Inspections) and a curated NASA-analogous dataset - strong for
  demonstrating behavior on authentic messy data, but not a substitute for an
  independently collected, domain-specific evaluation set in a final
  publication.
- Never write a performance percentage into the paper unless it was produced
  by an actual Research Evaluation run on the final dataset, with its seed,
  corruption level and metric definition recorded alongside it.
