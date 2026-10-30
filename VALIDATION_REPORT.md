# SelfHealDS — Validation Report

Regenerated after the bug-fix pass described in `CHANGELOG.md`. All numbers
below were produced by actually running the current codebase end to end
(`python -m pytest tests/` and direct calls into `backend/case_studies.py`,
`backend/research.py`), not copied from an earlier build.

## Test suite
28/28 tests pass (`python -m pytest tests/ -q`).

## Real-world case studies (detect -> repair -> post-repair validation)

| Case | Input rows | Cleaned rows | Detected findings | Repair events | Remaining Error-level | Review rows |
|---|---:|---:|---:|---:|---:|---:|
| Mars Climate Orbiter (unit mismatch, analogous dataset) | 1,015 | 1,000 | 2,187 | 2,346 | 0 | 0 |
| Hospital quality benchmark (HoloClean) | 183 | 183 | 301 | 189 | 0 | 0 |
| Flights multi-source consistency (HoloClean) | 803 | 800 | 1,134 | 884 | 0 | 0 |
| Chicago Food Inspections | 1,003 | 1,000 | 299 | 13 | 0 | 0 |
| UCI Air Quality sensor data | 1,200 | 1,200 | 1,243 | 1,224 | 0 | 0 |
| UCI Online Retail | 1,202 | 1,200 | 1,886 | 2,406 | 0 | 0 |
| UCI Adult / Census Income | 3,000 | 3,000 | 3,559 | 3,216 | 0 | 0 |

Row-count reduction is exclusively from exact-duplicate removal, never from
the validation gate silently discarding unresolved records — every case
finishes with 0 quarantined rows and 0 remaining Error-level findings, and
every cleaned CSV is non-empty.

Each case's main documented problem is confirmed detected and repaired, with
before -> repair -> after values visible in the app's "Key issue rows" panel:
- **NASA (unit mismatch):** mixed-unit measurements convert to canonical units; unrecognized tokens fall back to an observed-data median rather than a guessed conversion.
- **Hospital:** malformed provider identifiers, phone-number structure violations and benchmark inconsistencies are repaired without an empty output.
- **Flights:** invalid HHMM times, non-numeric delay values and textual boolean variants are repaired; genuinely noisy source-conflict fields (documented in the HoloClean paper) are correctly left for review rather than guessed.
- **Chicago:** malformed ZIP/date/risk/category values are repaired; exact duplicate inspection records are removed.
- **UCI Air Quality:** the documented `-200` missing-value sentinel is detected as an Error on the *raw* dataset, then converted to missingness and imputed under the declared policy.
- **UCI Online Retail:** legitimate negative cancellation quantities are correctly left untouched (per the dataset's own documented semantics); malformed quantity/date and negative unit-price values are repaired.
- **UCI Adult Census:** documented categorical placeholders and malformed age/work-hours values are repaired; `native-country` values such as `United-States` are now correctly recognized (see `CHANGELOG.md`, country-vocabulary fix).

These CSVs are locally supplied, documented public benchmarks (or, for NASA,
an explicitly labelled analogous dataset) — not the original historical
incident records. The app UI distinguishes documented/natural provenance from
controlled/synthetic corruption for every case.

## Semantic TXT pipeline

| Sample | Input lines | Findings before | Error-level before | Repair events | Error-level after |
|---|---:|---:|---:|---:|---:|
| 07_customer_support_semantic.txt | 6 | 9 | 3 | 4 | 0 |
| 08_employee_records_semantic.txt | 8 | 7 | 2 | 4 | 0 |
| 09_incident_log_semantic.txt | 7 | 8 | 4 | 5 | 0 |
| 10_product_notes_semantic.txt | 7 | 4 | 1 | 3 | 0 |

TXT processing uses a dedicated semantic pipeline: cross-sentence
contradictions for repeated entities, city/state/country contradictions,
contextual spelling errors, duplicate lines, obfuscated email formats,
whitespace inconsistencies, and placeholder detection. Automatic repair
requires deterministic reference evidence or repeated-document majority
evidence — ambiguous prose is never guessed. Every sample resolves to 0
Error-level findings after healing.

## CSV sample datasets

| Sample | Rows | Findings | Repair events | Cleaned output |
|---|---:|---:|---:|---|
| 01_customer_data_messy.csv | 6 | 22 | 32 | non-empty |
| 02_measurements_units.csv | 6 | 14 | 15 | non-empty |
| 03_employee_hr_messy.csv | 6 | 11 | 17 | non-empty |
| 04_ecommerce_transactions.csv | 6 | 9 | 16 | non-empty |
| 05_health_measurements_messy.csv | 5 | 14 | 14 | non-empty |
| 06_locations_semantic_errors.csv | 10 | 5 | 7 | non-empty |

## Research Evaluation module (labelled benchmark)
Example run on `01_customer_data_messy.csv`: precision 100%, recall 37.5%,
F1 54.6% (small demo file — 8 ground-truth cases; the small-sample warning
correctly fires). Example run on the Hospital case-study CSV (183 rows, real
benchmark): precision 100%, recall 43.2%, F1 60.4% over 37 injected cases.
Recall is intentionally moderate rather than inflated: roughly half of the
injected corruption mix is deliberately "plausible" (structurally valid but
numerically wrong), which the system is designed to *not* pretend to catch
without ground truth — see `RESEARCH_METHOD.md`, "abstention boundary."

Full sensitivity sweep, ablation, and baseline-comparison tables are
generated live in the app's Research Evaluation page and exported as a
timestamped, fingerprinted JSON manifest — see `POSITIONING.md` for how to
read them and `CHANGELOG.md` for the two evaluation-methodology bugs fixed in
this pass (camelCase column coverage; pre-existing-finding double subtraction).

## LLM-comparison baseline
`backend/llm_baseline.py` (wired into the Research Evaluation page) sends the
same corrupted-benchmark file to a general-purpose model and scores it with
identical metrics. Its scoring logic is unit-tested against a mocked network
layer in `tests/test_llm_baseline.py` (4/4 passing) — this confirms the
comparison arithmetic is correct, but the module has **not** been exercised
against a live model call in this environment, since no API key is available
here. Run it yourself with your own key before citing a number from it.

