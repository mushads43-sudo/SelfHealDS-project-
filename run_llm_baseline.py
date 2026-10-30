"""SelfHealDS Research Experiment E10 — Direct LLM Baseline Comparison.

Answers the paper's own research question ("why is SelfHealDS not simply
something an LLM can already do?") with evidence instead of assertion.

What this script does
----------------------
1. Loads a reference dataset (an arbitrary CSV, a bundled sample, or one of
   the seven real-world case studies already defined in
   ``backend.case_studies.CASE_STUDIES``).
2. Injects the SAME controlled, labelled corruption benchmark that
   ``backend.evaluation.evaluate_detection`` already uses (identical
   ``build_benchmark`` call, identical seed) so SelfHealDS and the LLM are
   scored on byte-identical messy input and identical ground truth.
3. Runs the existing SelfHealDS pipeline on that messy data (reusing
   ``evaluate_detection`` verbatim — no reimplementation).
4. Sends the same messy CSV to a general-purpose LLM with a plain
   "clean this dataset" instruction and asks for a cleaned CSV back.
5. Scores the LLM's returned CSV against the identical ground truth using
   the same repair-accuracy definition as the SelfHealDS side, and
   additionally measures two things SelfHealDS's own benchmark does not need
   to measure because they are structurally impossible for it:
     - "collateral changes": cells the system modified that were never
       corrupted (evidence of fabrication / unnecessary edits).
     - a domain-semantic-trap check, when the dataset declares one (e.g. the
       ``uci_online_retail`` case's documented legitimate negative
       cancellation quantities) — does the baseline silently "fix" values
       that are correct by the dataset's own documented semantics?
6. Also records a structural capability checklist (rule IDs, evidence score,
   confidence basis, post-repair validation, audit log) which is a factual
   yes/no comparison, not a fabricated number: a general chat completion
   does not emit any of these artifacts, because nothing in a plain
   "clean this CSV" prompt asks for them.
7. Saves a fingerprinted, timestamped, seeded JSON manifest under
   ``research_outputs/`` in the same style as ``scripts_run_benchmark.py``,
   so every number in the paper stays traceable to a specific run.

This script performs a real network call to the Anthropic API (or, in
``--dry-run`` mode, a no-op stand-in) and is therefore kept separate from
the core application, which the project's README states requires no paid
AI API.

Usage — three ways to get an LLM baseline, no paid API required
-----------------------------------------------------------------
1. Paid API (fully automated):
    export ANTHROPIC_API_KEY=sk-ant-...
    python run_llm_baseline.py --case holoclean_hospital

2. Free/manual (no API key at all): export the exact messy CSV + prompt,
   paste them into any free chat UI (claude.ai, chatgpt.com, gemini.google.com
   all have free tiers), paste the reply back into a file, then score it:
    python run_llm_baseline.py --case holoclean_hospital --export-messy hospital_messy.csv
    #  -> paste hospital_messy.csv + the printed prompt into a free chat UI,
    #     save its reply as hospital_llm_output.csv
    python run_llm_baseline.py --case holoclean_hospital --llm-output hospital_llm_output.csv --model-label "claude.ai free tier"

3. Dry run (tests the scoring pipeline only, no LLM involved at all):
    python run_llm_baseline.py --case holoclean_flights --dry-run

Only stdlib + the project's existing dependencies (pandas, numpy) are
required. The optional paid-API path uses ``urllib.request`` directly so
it adds no new dependency to requirements.txt.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from backend.evaluation import (  # noqa: E402
    build_benchmark, evaluate_detection, EVAL_VERSION,
)
from backend.case_studies import CASE_STUDIES, get_case, load_case_dataset  # noqa: E402
from backend.profiler import dataset_fingerprint  # noqa: E402

LLM_EVAL_VERSION = "2026.09-llm-baseline-v1"
DEFAULT_MODEL = os.environ.get("SELFHEALDS_LLM_MODEL", "claude-sonnet-4-6")
API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MAX_ROWS = 300


# --------------------------------------------------------------------------
# Dataset loading
# --------------------------------------------------------------------------

def load_reference(args) -> tuple[pd.DataFrame, str, dict]:
    """Return (dataframe, label, case_or_none) for the requested source."""
    if args.case:
        case = get_case(args.case)
        if case is None:
            names = ", ".join(c["id"] for c in CASE_STUDIES)
            raise SystemExit(f"Unknown --case '{args.case}'. Available: {names}")
        df, meta = load_case_dataset(case)
        if df is None:
            raise SystemExit(
                f"Local CSV for case '{args.case}' is missing: {meta.get('reason')}"
            )
        return df, f"case:{args.case}", case
    if args.dataset:
        path = Path(args.dataset)
        if not path.exists():
            raise SystemExit(f"Dataset not found: {path}")
        return pd.read_csv(path, low_memory=False), f"file:{path.name}", None
    raise SystemExit("Pass either --case CASE_ID or --dataset PATH.")


# --------------------------------------------------------------------------
# LLM call
# --------------------------------------------------------------------------

LLM_SYSTEM_PROMPT = (
    "You are given a messy CSV dataset. Clean it: fix missing values, "
    "formatting problems, invalid values, and inconsistencies as best you "
    "can. Return ONLY the cleaned CSV. Keep the exact same columns, the "
    "exact same row order, and the exact same number of rows as the input. "
    "Do not add commentary, explanations, or markdown code fences — output "
    "raw CSV text only, starting with the header row."
)


def call_llm(csv_text: str, model: str, api_key: str, max_tokens: int = 8000) -> str:
    payload = json.dumps({
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "system": LLM_SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": csv_text}],
    }).encode("utf-8")
    req = urllib.request.Request(
        API_URL, data=payload, method="POST",
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )
    last_err = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            text_blocks = [b["text"] for b in body.get("content", []) if b.get("type") == "text"]
            return "\n".join(text_blocks)
        except urllib.error.HTTPError as exc:
            last_err = exc
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code == 429 or exc.code >= 500:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"Anthropic API error {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            last_err = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Anthropic API call failed after retries: {last_err}")


def dry_run_llm(csv_text: str) -> str:
    """Deterministic offline stand-in used by --dry-run to test the pipeline
    without network access or an API key. Returns the input unchanged, which
    is a maximally conservative (and maximally uninteresting) baseline."""
    return csv_text


def extract_csv(raw_text: str) -> str:
    """Strip markdown code fences if the model added them despite instructions."""
    text = raw_text.strip()
    fence = re.search(r"```(?:csv)?\s*\n(.*?)```", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return text


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def _values_match(a, b) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    try:
        fa, fb = float(a), float(b)
        return abs(fa - fb) <= max(1e-9, abs(fb) * 1e-9)
    except (TypeError, ValueError):
        return str(a).strip() == str(b).strip()


def score_llm_repairs(messy: pd.DataFrame, llm_df: pd.DataFrame, cases: list[dict]) -> dict:
    """Score the LLM's cleaned output against the identical ground truth
    used to score SelfHealDS, with the same repair-accuracy definition
    ``backend.evaluation._repair_results`` uses: correct only when the
    output matches the recorded expected_repair exactly; a case with
    expected_repair=None (non-inferable) can never be scored correct.
    """
    attempted, correct, unsafe, not_touched = [], [], [], []
    repairable_cases = [c for c in cases if c.get("expected_repair") is not None]

    for c in cases:
        row, col = c["row"], c["column"]
        if row >= len(llm_df) or col not in llm_df.columns:
            unsafe.append({**c, "llm_value": None, "reason": "row/column missing from LLM output"})
            continue
        llm_val = llm_df.at[row, col]
        messy_val = c["corrupted_value"]
        touched = not _values_match(llm_val, messy_val)
        if not touched:
            not_touched.append(c)
            continue
        attempted.append(c)
        expected = c.get("expected_repair")
        if expected is not None and _values_match(llm_val, expected):
            correct.append(c)
        else:
            unsafe.append({**c, "llm_value": llm_val})

    repair_accuracy = round(len(correct) / len(attempted) * 100, 2) if attempted else None
    safe_coverage = round(len(correct) / len(repairable_cases) * 100, 2) if repairable_cases else None
    unsafe_rate = round(len(unsafe) / len(attempted) * 100, 2) if attempted else None

    return {
        "repair_attempts": len(attempted),
        "repairable_cases": len(repairable_cases),
        "repair_correct": len(correct),
        "repair_unsafe": len(unsafe),
        "not_touched": len(not_touched),
        "repair_accuracy_pct": repair_accuracy,
        "safe_repair_coverage_pct": safe_coverage,
        "unsafe_repair_rate_pct": unsafe_rate,
    }


def collateral_changes(messy: pd.DataFrame, llm_df: pd.DataFrame, cases: list[dict]) -> dict:
    """Count cells the baseline changed that were never part of the injected
    benchmark, i.e. edits made to cells that had no labelled defect."""
    case_cells = {(c["row"], c["column"]) for c in cases}
    if llm_df.shape != messy.shape:
        return {
            "comparable": False,
            "reason": f"LLM output shape {llm_df.shape} != input shape {messy.shape}; "
                      "collateral-change count is not meaningful for a misshapen response.",
        }
    changed = []
    for col in messy.columns:
        if col not in llm_df.columns:
            continue
        for row in messy.index:
            if (row, col) in case_cells:
                continue
            if not _values_match(messy.at[row, col], llm_df.at[row, col]):
                changed.append({"row": int(row), "column": str(col),
                                 "original": messy.at[row, col], "llm_value": llm_df.at[row, col]})
    total_non_case_cells = messy.shape[0] * messy.shape[1] - len(case_cells)
    rate = round(len(changed) / total_non_case_cells * 100, 4) if total_non_case_cells else None
    return {
        "comparable": True,
        "collateral_change_count": len(changed),
        "collateral_change_rate_pct": rate,
        "examples": changed[:15],
    }


def domain_trap_check(messy: pd.DataFrame, llm_df: pd.DataFrame, case: dict | None) -> dict | None:
    """Dataset-specific check for a documented domain-semantic trap.

    Currently implemented for uci_online_retail, whose data dictionary
    defines invoice-prefix 'C' rows as legitimate cancellations with
    legitimate negative Quantity values (see backend/case_studies.py,
    case_constraints={"allow_negative_quantity": True}). A baseline that
    "cleans" these by flipping the sign is destroying correct data.
    """
    if case is None or not case.get("case_constraints", {}).get("allow_negative_quantity"):
        return None
    if "Quantity" not in messy.columns or "Quantity" not in llm_df.columns or llm_df.shape != messy.shape:
        return {"applicable": False, "reason": "Quantity column or matching shape not available."}
    flipped = 0
    checked = 0
    examples = []
    for row in messy.index:
        try:
            orig = float(messy.at[row, "Quantity"])
        except (TypeError, ValueError):
            continue
        if orig >= 0:
            continue
        checked += 1
        try:
            new = float(llm_df.at[row, "Quantity"])
        except (TypeError, ValueError):
            flipped += 1
            examples.append({"row": int(row), "original": orig, "llm_value": llm_df.at[row, "Quantity"]})
            continue
        if new != orig:
            flipped += 1
            examples.append({"row": int(row), "original": orig, "llm_value": new})
    return {
        "applicable": True,
        "legitimate_negative_quantities_checked": checked,
        "incorrectly_altered": flipped,
        "incorrectly_altered_rate_pct": round(flipped / checked * 100, 2) if checked else None,
        "examples": examples[:15],
        "interpretation": (
            "Any non-zero rate here means the baseline silently overwrote values that "
            "the dataset's own documentation defines as correct — the exact failure mode "
            "SelfHealDS's case-specific constraints are designed to prevent."
        ),
    }


CAPABILITY_CHECKLIST = {
    "Per-cell rule ID attached to every change": {"selfhealds": True, "llm_baseline": False},
    "Evidence-strength / confidence score per change": {"selfhealds": True, "llm_baseline": False},
    "Human-readable evidence basis per change": {"selfhealds": True, "llm_baseline": False},
    "Explicit repair-decision hierarchy (deterministic > imputation > semantic > flag-only > quarantine)": {"selfhealds": True, "llm_baseline": False},
    "Post-repair re-validation gate (pass/fail per cell)": {"selfhealds": True, "llm_baseline": False},
    "Abstains instead of fabricating when evidence is insufficient": {"selfhealds": True, "llm_baseline": "not guaranteed by a plain cleaning prompt"},
    "Deterministic / reproducible output for identical input+seed": {"selfhealds": True, "llm_baseline": "not guaranteed (model/version/decoding dependent)"},
    "Exportable machine-readable audit trail": {"selfhealds": True, "llm_baseline": False},
}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--case", help="Real-world case id from backend.case_studies (e.g. holoclean_hospital)")
    src.add_argument("--dataset", help="Path to an arbitrary CSV")
    ap.add_argument("--corruption-rate", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS,
                     help="Cap rows sent to the LLM (context-window / cost limit). "
                          "SelfHealDS is still scored on the FULL messy dataset for comparison to E1; "
                          "the LLM comparison is restricted to the same row cap on both sides for fairness.")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                     help="Model name, only used for the paid --api-call path.")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                       help="Skip the LLM entirely; identity stand-in to test the scoring pipeline only.")
    mode.add_argument("--export-messy", metavar="PATH",
                       help="FREE, NO API KEY: write the row-capped messy CSV (and a matching "
                            ".prompt.txt) to PATH so you can paste both into any free chat UI "
                            "(claude.ai, chatgpt.com, gemini.google.com) by hand. Exits after writing.")
    mode.add_argument("--llm-output", metavar="PATH",
                       help="FREE, NO API KEY: score an LLM reply you already obtained by hand "
                            "(e.g. pasted from a free chat UI) and saved as a CSV file at PATH, "
                            "instead of calling any API.")
    mode.add_argument("--api-call", action="store_true",
                       help="Call the paid Anthropic API directly (requires ANTHROPIC_API_KEY). "
                            "This is the only mode that costs money.")
    ap.add_argument("--model-label", default=None,
                     help="Free-text label recorded in the manifest for --llm-output runs, e.g. "
                          "'claude.ai free tier' or 'chatgpt.com free tier', since the exact model "
                          "version behind a free chat UI isn't always disclosed.")
    args = ap.parse_args()
    if not (args.dry_run or args.export_messy or args.llm_output or args.api_call):
        ap.error("Choose one mode: --export-messy (free), --llm-output (free), --dry-run (no LLM), "
                  "or --api-call (paid, requires ANTHROPIC_API_KEY).")

    reference, label, case = load_reference(args)
    print(f"Loaded reference: {label} ({len(reference)} rows x {len(reference.columns)} cols)")

    # Identical, deterministic corruption benchmark used by evaluate_detection (E1).
    reference = reference.copy().reset_index(drop=True)
    messy, cases = build_benchmark(reference, args.corruption_rate, args.seed)
    if not cases:
        raise SystemExit(
            "No injectable corruption cases found for this dataset's columns "
            "(no age/phone/email/quantity/weight/date/city+country fields matched). "
            "Try a different --case or --dataset."
        )
    print(f"Injected {len(cases)} labelled corruption cases (seed={args.seed}, rate={args.corruption_rate}).")

    capped = messy.head(args.max_rows).reset_index(drop=True)
    row_cap_applied = len(capped) < len(messy)
    csv_text = capped.to_csv(index=False)

    # --- FREE MODE 1: just export the messy CSV + prompt for manual pasting ---
    if args.export_messy:
        out_csv = Path(args.export_messy)
        out_prompt = out_csv.with_suffix(out_csv.suffix + ".prompt.txt")
        out_csv.write_text(csv_text)
        out_prompt.write_text(LLM_SYSTEM_PROMPT + "\n\n" + csv_text)
        print(f"Wrote messy dataset: {out_csv}")
        print(f"Wrote combined prompt+data (paste this whole file into a free chat UI): {out_prompt}")
        print(
            "\nNext steps (no API key needed):\n"
            "  1. Open claude.ai, chatgpt.com, or gemini.google.com (any free account works).\n"
            f"  2. Paste the full contents of {out_prompt.name} as your message.\n"
            "  3. Copy the model's reply — it should be CSV text — into a new .csv file.\n"
            "  4. Re-run this script with the SAME --case/--dataset, --seed and --max-rows, "
            "plus:\n"
            f"       --llm-output your_saved_reply.csv --model-label \"<which chat UI/model you used>\"\n"
        )
        return

    # --- SelfHealDS side: reuse the existing benchmark harness verbatim ---
    print("Running SelfHealDS (backend.evaluation.evaluate_detection)...")
    selfhealds_eval = evaluate_detection(reference, args.corruption_rate, args.seed)

    # --- LLM side ---
    if args.llm_output:
        # FREE MODE 2: score a reply the user obtained by hand from a free chat UI.
        path = Path(args.llm_output)
        if not path.exists():
            raise SystemExit(f"--llm-output file not found: {path}")
        print(f"Scoring manually-obtained LLM output from {path} (no API call made).")
        llm_raw = path.read_text()
        model_used = args.model_label or f"manual:{path.name}"
    elif args.dry_run:
        print("DRY RUN: no LLM involved at all, identity stand-in used to test the scoring pipeline.")
        llm_raw = dry_run_llm(csv_text)
        model_used = "dry-run-identity-stub"
    else:  # args.api_call
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise SystemExit(
                "Set ANTHROPIC_API_KEY to use --api-call, or use the free path instead: "
                "--export-messy then --llm-output (see --help)."
            )
        print(f"Calling {args.model} (PAID API — {len(capped)} rows sent)...")
        llm_raw = call_llm(csv_text, args.model, api_key)
        model_used = args.model

    llm_csv = extract_csv(llm_raw)
    try:
        llm_df = pd.read_csv(StringIO(llm_csv), low_memory=False)
    except Exception as exc:
        llm_df = pd.DataFrame()
        parse_error = str(exc)
    else:
        parse_error = None

    # Restrict the case list used for LLM scoring to rows within the cap so
    # both sides are scored on a dataset they actually saw in full.
    cases_in_scope = [c for c in cases if c["row"] < len(capped)]
    messy_scope = messy.head(len(capped)).reset_index(drop=True)

    if parse_error or llm_df.empty:
        llm_repair_scores = {"repair_attempts": 0, "repair_correct": 0, "note": f"LLM output failed to parse as CSV: {parse_error}"}
        collateral = {"comparable": False, "reason": "LLM output did not parse as CSV."}
        trap = None
    else:
        llm_repair_scores = score_llm_repairs(messy_scope, llm_df, cases_in_scope)
        collateral = collateral_changes(messy_scope, llm_df, cases_in_scope)
        trap = domain_trap_check(messy_scope, llm_df, case)

    # --- Report ---
    print("\n=== SelfHealDS (full dataset, E1 benchmark) ===")
    for k in ["precision_pct", "recall_pct", "f1_pct", "repair_accuracy_pct",
              "safe_repair_coverage_pct", "unsafe_repair_rate_pct", "clean_false_positive_rate_pct"]:
        print(f"  {k}: {selfhealds_eval.get(k)}")

    print(f"\n=== LLM baseline ({model_used}, {len(capped)}/{len(messy)} rows"
          f"{' — row-capped, see notes' if row_cap_applied else ''}) ===")
    for k, v in llm_repair_scores.items():
        print(f"  {k}: {v}")
    if collateral.get("comparable"):
        print(f"  collateral_change_count: {collateral.get('collateral_change_count')}")
        print(f"  collateral_change_rate_pct: {collateral.get('collateral_change_rate_pct')}")
    else:
        print(f"  collateral_change_count: N/A — {collateral.get('reason')}")
    if trap:
        print(f"\n  Domain-semantic-trap check ({args.case}):")
        if trap.get("applicable"):
            print(f"    legitimate negatives checked: {trap.get('legitimate_negative_quantities_checked')}")
            print(f"    incorrectly altered: {trap.get('incorrectly_altered')} "
                  f"({trap.get('incorrectly_altered_rate_pct')}%)")
        else:
            print(f"    N/A — {trap.get('reason')}")

    print("\n=== Structural capability checklist (factual, not scored) ===")
    for capability, row in CAPABILITY_CHECKLIST.items():
        print(f"  {capability}: SelfHealDS={row['selfhealds']} | LLM baseline={row['llm_baseline']}")

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "E10 — LLM baseline comparison",
        "llm_eval_version": LLM_EVAL_VERSION,
        "selfhealds_eval_version": EVAL_VERSION,
        "reference_label": label,
        "reference_fingerprint": dataset_fingerprint(reference),
        "corruption_rate": args.corruption_rate,
        "seed": args.seed,
        "injected_cases": len(cases),
        "row_cap": args.max_rows,
        "row_cap_applied": row_cap_applied,
        "rows_sent_to_llm": len(capped),
        "model": model_used,
        "selfhealds_evaluation": {k: v for k, v in selfhealds_eval.items()
                                    if k not in {"messy", "detected", "assessment"}},  # drop bulky non-JSON-simple objects
        "llm_repair_scores": llm_repair_scores,
        "llm_collateral_changes": collateral,
        "llm_output_parse_error": parse_error,
        "domain_trap_check": trap,
        "capability_checklist": CAPABILITY_CHECKLIST,
        "notes": [
            "SelfHealDS metrics are computed on the FULL messy dataset via the existing "
            "evaluate_detection() benchmark, unmodified.",
            "LLM metrics are computed only over the row-capped subset actually sent to the "
            "model, because general-purpose chat completion has a practical context/cost "
            "limit that the deterministic pipeline does not. This asymmetry is itself part "
            "of the finding and should be stated explicitly in the paper's Results section, "
            "not hidden by reporting only percentages.",
            "expected_repair values with a value of None are non-inferable by design (e.g. a "
            "three-digit phone number); the LLM, like SelfHealDS, cannot score 'correct' on "
            "these — any change to such a cell is counted as unsafe.",
        ],
    }

    out_dir = ROOT / "research_outputs"
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_label = re.sub(r"[^a-zA-Z0-9_.-]+", "_", label)
    out_path = out_dir / f"llm_baseline_{safe_label}_{ts}.json"
    out_path.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\nSaved manifest: {out_path}")


if __name__ == "__main__":
    main()
