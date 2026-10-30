# SelfHealDS — Changelog (Critical Review Pass)

This pass audited the full codebase (backend + `app.py` + tests + docs),
ran every feature end to end against real data, and fixed the concrete bugs
found. All 32 tests pass (28 original + 4 new for the LLM-baseline module)
and all 7 real-world case studies, 4 TXT samples and 6 CSV samples were
re-verified after every change.

## Correctness bugs fixed

1. **Python-version portability crash (`app.py`).** Two download-button calls
   used Python-3.12-only nested-same-quote f-string syntax
   (`f"{case["local_filename"]...}"`), which raises `SyntaxError` on any
   Python <=3.11 — i.e. the app would fail to even start in most real
   deployment/grading environments. Rewritten to plain string ops.

2. **Semantic-type substring false positives (`backend/rules.py`).**
   `semantic_type()` matched column names by raw substring, so a column named
   `Destination` matched the substring `"nation"` and was silently treated as
   a *country* field — flagging 803 airport codes (`ORD`, `ATL`, ...) as
   "Unknown country" in the Flights case study. Same class of bug risked
   misfiring on any column containing `"age"` (`Average`, `Package`,
   `Message`, ...), `"date"` (`Update`, `Candidate`, ...), etc. Fixed by
   switching to whole-token matching plus camelCase-boundary-aware column
   normalization (`ProviderNumber` -> tokens `{provider, number}`).

3. **Undersized country reference vocabulary (`backend/rules.py`,
   `backend/detector.py`).** Only 14 countries were recognized, so common,
   entirely legitimate values — `Mexico`, `Netherlands`, hyphenated
   `United-States` (the actual UCI Adult Census format) — were flagged as
   errors. This alone produced 1,800+ false "Unknown country" findings on the
   3,000-row Adult Census case study. Fixed by adding a ~195-entry world
   country reference list (kept separate from the small, intentionally
   curated city<->country demo mapping used for cross-field consistency
   checks) plus hyphen/underscore-aware token normalization.

4. **Duplicate/contradictory findings on placeholder cells (`backend/detector.py`,
   `backend/rules.py`).** A placeholder token (`"?"`, `"n/a"`, ...) in a
   typed column (phone/email/date/country/measurement) was flagged *twice*:
   once correctly as "Placeholder value" (a missingness finding) and once
   misleadingly as an unrelated format/domain error for the same cell (e.g.
   `?` in a country column also produced "Unknown country"). Fixed with a
   shared `is_placeholder_token()` guard applied consistently across all
   type-specific detector loops, so each placeholder cell is now reported
   exactly once, under the correct dimension.

5. **Evaluation-methodology bug: silent recall undercounting
   (`backend/evaluation.py`).** The benchmark subtracted the *entire*
   pre-existing findings set from newly detected findings before scoring. If
   a corruption was injected into a cell that already had a pre-existing
   finding of the *same issue type* (which happens on real, not-yet-clean
   reference data by design), the post-injection detection was silently
   discarded as "not new" — even though the detector correctly caught the
   injected corruption. Verified concretely on `06_locations_semantic_errors.csv`:
   recall for a city/country injection went from an incorrect 0% to the
   correct 100% after the fix. Fixed by excluding only the specific injected
   `(row, column)` cells from the pre-existing-findings subtraction, in
   `evaluate_detection`, `compare_with_baseline` and `run_ablation`.

6. **Evaluation coverage gap on camelCase datasets (`backend/evaluation.py`).**
   The benchmark's own column-type matcher (`_has_token`) used the same
   no-camelCase-splitting logic as bug #2, so it silently produced **zero
   candidate corruption specs** on any camelCase-headered dataset —
   concretely, running the in-app Research Evaluation page on the Hospital
   case study CSV (`PhoneNumber`, `ProviderNumber`, ...) returned
   precision/recall/F1 of `None` across the board, looking like a broken
   feature. Fixed with the same camelCase-aware tokenization; verified the
   Hospital, Flights and Chicago datasets now produce real, non-degenerate
   metrics (e.g. Hospital: precision 100%, recall 43.2%, F1 60.4% over 37
   injected cases).

## Documentation consolidated
Nine overlapping markdown files (leftover from iterative build history) were
consolidated into four, removing duplicated positioning/methodology content
per the presentation-readiness requirement:
- `RESEARCH_METHOD.md` <- merged `RESEARCH_METHOD.md` + `RESEARCH_METHODOLOGY.md`
- `POSITIONING.md` (new) <- merged `RESEARCH_GRADE_POSITIONING.md` + `FINAL_PRESENTATION_GUIDE.md` + `FINAL_UI_RESEARCH_GRADE.md`
- `VALIDATION_REPORT.md` <- merged `REAL_WORLD_VALIDATION_REPORT.md` + `TEXT_FILE_SUPPORT_VALIDATION.md`, numbers regenerated post-fix
- `RESEARCH_GRADE_CHANGES.md` retired; its content is superseded by this file
- `README.md` and `ARCHITECTURE.md` kept as-is (no meaningful duplication with the above)

## What was intentionally left unchanged
The core architecture (deterministic rule engine + Isolation Forest hybrid,
no LLM), the UI navigation structure, the "Why SelfHealDS?" differentiation
argument, and the research evaluation harness design were already sound and
are described accurately in `POSITIONING.md` and `RESEARCH_METHOD.md`. This
pass fixed implementation bugs in that design, not the design itself.

## UI presentation pass (this session)
7. **Text size / spacing / hierarchy.** Base font size raised (`app.py`
   CSS block) and every sub-12px label/caption bumped up a tier; content
   width and padding increased to use screen space better on wide displays.
   Verified `app.py` still parses and all tests still pass after the change.

8. **Removed duplicated UI messaging.** The "Evidence / Safety /
   Traceability / Reproducibility" pill ribbon and the "Guarded Healing
   Loop" workflow banner previously rendered at full size on *every* page
   (Rule Catalogue, Data Contract, Reports & Export, ...) — 11 repeats of
   the same pitch per session. They now render at full size only where they
   orient the reader (Overview, Auto-Heal, Why SelfHealDS?); every other
   page keeps a slim one-line breadcrumb instead.

9. **Standardized the workflow language to the requested 7 steps** —
   Detect -> Explain -> Root cause -> Repair -> Re-check -> Validate -> Prove
   — everywhere it appears (it had drifted to two different 6-step wordings
   in different places: "...Verify -> Prove" in one banner and
   "...Re-check -> Accept/Review" on the Why SelfHealDS page).

10. **Plain-language glossary lines added** ahead of the densest Research
    Evaluation subsections (baseline comparison, ablation, corruption-level
    sensitivity, error-category performance, sample-suite comparison), and
    the "Clean FPR" metric was renamed "False-alarm rate" (with the
    technical term kept in the definition tooltip) so a non-technical reader
    is not left decoding acronyms to follow the story.

## LLM-comparison baseline (this session)
11. **Built a real, working, opt-in LLM-cleaning comparison**
    (`backend/llm_baseline.py`, wired into the Research Evaluation page).
    It sends the identical injected-corruption benchmark file used to score
    SelfHealDS to a general-purpose model via a single cleaning prompt, and
    scores the response with the same precision/recall/F1/repair-accuracy
    methodology — the strongest available empirical version of "why not just
    an LLM," rather than an assertion. It requires the researcher's own API
    key (session-only, never persisted) and never runs automatically, so it
    does not weaken the core pipeline's "local-first, no paid API required"
    property. Covered by `tests/test_llm_baseline.py` (4 tests, network layer
    mocked) — it has not been exercised against a live model in this sandbox
    because no API credentials are available here; this is disclosed, not
    hidden, in `POSITIONING.md`.

## Known remaining limitation (not fixed — flagged, not hidden)
`backend/evaluation.py`'s corruption-injection framework occasionally
produces a genuinely low-signal benchmark on datasets whose only eligible
semantic column has near-zero detectable "obvious" corruption headroom
(e.g. a 6-row unit-measurement demo file). This is expected/documented
behavior for small samples (the app's own "small-sample warning" correctly
fires), not a bug — but a paper submission should run the benchmark only on
datasets that clear the `MIN_RESEARCH_ROWS` (30-row) threshold, as the app
already recommends.
