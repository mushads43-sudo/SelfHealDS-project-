# SelfHealDS — Positioning, Differentiation and Demo Guide

## One sentence
SelfHealDS is an evidence-driven, guarded data-quality pipeline that detects
corruption, explains why it is wrong, identifies a likely root cause, repairs
only when the evidence supports it, re-checks the result, and preserves a
complete audit trail — entirely deterministic, with no LLM in the loop.

## The visible workflow
Detect -> Explain -> Root cause -> Repair -> Re-check -> Validate -> Prove

## What the system actually is
A local-first pipeline combining structured validation (rules + regex +
domain constraints), semantic reasoning (city/country, cross-record
consistency, cross-sentence text contradictions), unit-aware normalization, a
statistical anomaly layer (Isolation Forest, detection-only), guarded repair
authorization, post-repair re-validation, and full repair provenance. No step
calls a paid or unpaid AI/LLM API — every decision is reproducible from the
declared rule set and a fixed random seed.

## The honest research differentiator
Do not claim "no previous system or AI has ever cleaned data" — that claim is
false and indefensible. Individual techniques here (regex/domain validation,
median imputation, Isolation Forest, unit conversion, and LLM-assisted
cleaning generally) already exist elsewhere. **The defensible contribution is
the integrated, evidence-driven closed loop and its measurable safety and
evaluation protocol** — not any single operation in isolation.

### Why this is more than "ask an AI to clean the file"
| A general-purpose AI/LLM | SelfHealDS |
|---|---|
| May propose a plausible-looking fix from a prompt | Records the *specific rule and observed evidence* that justified each change |
| Output can vary between runs/models/temperature | Deterministic: same input + same ruleset + same seed -> same output, every time |
| No standard mechanism to say "I don't know" per-cell | Explicit do-no-harm boundary: insufficient evidence -> quarantined for review, never guessed |
| Cleaned file is typically the only artifact | Re-scans its own output and marks any repair that still fails validation |
| Rationale, if given, is prose, not machine-checkable | Every change ships with rule ID, evidence, confidence-strength, before/after value, validation status — exportable as JSON |
| No inherent measurement of correctness | Ships its own benchmark harness: precision/recall/F1, repair accuracy, false-repair rate, ablation, baseline comparison |
| Correcting a systemic error means re-prompting and hoping | Correcting a systemic error means editing one rule/policy value, which is versioned and auditable |

This is a claim about **process guarantees (reproducibility, evidence,
abstention, auditability)**, not a claim that the underlying detection logic
is unprecedented. A reviewer should come away thinking "this is a governed
quality-control protocol you could put a compliance sign-off on," not "this
just wraps an LLM prompt."

## Six differentiating capabilities (and where to see them in the app)
1. **Evidence-backed repair** — every automatic transformation records rule,
   evidence, confidence-strength and replacement. *Explainability page.*
2. **Root-cause view** — related findings are grouped so downstream symptoms
   (e.g. a statistical outlier) can be linked to an upstream cause (e.g. a
   unit or semantic inconsistency). *Auto-Heal -> Root-cause view.*
3. **Do-no-harm boundary** — high-confidence repairs are automated; uncertain
   cases are quarantined for review instead of fabricated. *Every repair log.*
4. **Closed-loop validation** — the repaired data is independently re-scanned;
   a repair that still leaves an Error at the same cell is marked FAILED, not
   silently accepted. *Auto-Heal -> validation banner.*
5. **Full provenance** — original value, new value, rule, reason, confidence
   and validation status are recorded per change and exportable as JSON.
   *Audit Trail, Reports & Export.*
6. **Reproducible evaluation** — dataset fingerprint, ruleset version, seeds,
   corruption protocol and baseline are retained with every metric.
   *Research Evaluation page.*

## Demo script (live walkthrough)
1. Open a Real-World Case Study (e.g. UCI Air Quality) — show the documented
   incident and provenance, not just a file upload.
2. Show BEFORE metrics and the raw findings table.
3. Open Explainability on one concrete finding — show rule, evidence,
   confidence basis and proposed treatment.
4. Run Auto-Heal — show BEFORE -> AFTER error counts, the validation gate
   result, and any quarantined rows.
5. Open the repair audit table — original value -> repaired value -> rule ->
   evidence -> validation status, for the specific case-study issue.
6. Open Research Evaluation — run the benchmark live; show precision/recall/F1,
   repair accuracy, ablation and baseline comparison generated in real time
   (not hard-coded numbers).
7. Close on "Why SelfHealDS?" and the comparison table above.

## In-app navigation map
```
START       Sample Datasets | Real-World Case Studies | Upload Your Own Dataset
UNDERSTAND  Overview | Data Profile | Data Assessment
HEAL&PROVE  Explainability | Auto-Heal | Audit Trail
GOVERN      Rule Catalogue | Data Contract
RESEARCH    Research Evaluation | Reports & Export | Why SelfHealDS?
```
The Overview page surfaces the four differentiators (Evidence, Safety,
Traceability, Reproducibility) immediately, without requiring the presenter
to explain the backend. The Auto-Heal page visibly shows the before -> after
validation loop with a pass/fail banner.

## Paper evaluation plan
Use independently sourced datasets where possible, distinguished from the
bundled controlled demo samples (the UI already separates "documented/natural"
from "controlled/synthetic" provenance for every case study). Report:
detection precision/recall/F1 on labelled corruptions; repair accuracy where
an independent target exists; false-repair rate on cells that were not
corrupted; post-repair Error-level findings; review/abstention and
unsafe-repair rates; runtime and throughput; ablation of deterministic rules
vs. the hybrid detector; robustness across corruption levels and seeds.

## LLM baseline comparison (implemented, opt-in)
The Research Evaluation page includes a working "compare against a
general-purpose LLM" section (`backend/llm_baseline.py`): it sends the exact
same corrupted-benchmark file used to score SelfHealDS to a general-purpose
model with a single cleaning prompt, and scores the model's response with the
identical precision/recall/F1/repair-accuracy methodology. This is the
strongest empirical version of the "why not just an LLM" claim — a live,
same-conditions number instead of an assertion.

This feature requires the researcher's own Anthropic API key (entered in the
UI as a password field, kept only for that session, never written to disk;
or supplied via the `ANTHROPIC_API_KEY` environment variable) and is never
called automatically — the core SelfHealDS pipeline remains fully local and
requires no paid API, which is a load-bearing part of this project's
positioning and is unaffected by this optional comparison tool. It was built
and unit-tested against a mocked network layer
(`tests/test_llm_baseline.py`); it has not been exercised against a live
model in this environment because no API credentials are available here.
Before citing a number from it in a paper, run it yourself and record the
model name, prompt, seed and corruption level alongside the result — treat a
single prompt/model/temperature configuration as one point in a large design
space, not a definitive verdict on what LLMs can do with more engineering
(few-shot examples, structured-output tools, multiple passes, self-critique).

Do not present illustrative or synthetic numbers as final paper results. Run
the Research Evaluation module on the final datasets and export its manifest.
