"""Build paper Tables T1-T8 from REAL SelfHealDS project data.

T1 and T2 need no experiment run at all — they come straight from the
project's own case-study catalog and rule/policy code, so this script always
produces them.

T3, T4, T5, T6, T7 need results from backend.research.run_research_suite()
(the same function the "Research Evaluation" page's "Run research benchmark"
button calls) and backend.pipeline.heal() (the same function "Auto-Heal"
calls). This script can EITHER run those directly on a local CSV (bypassing
the Streamlit UI entirely, so you don't have to click through the app) OR
read an already-downloaded selfhealds_research_run.json manifest.

T8 needs the JSON manifest(s) produced by run_llm_baseline.py.

Usage
-----
Always available (no experiment needed):
    python build_tables.py --t1 --t2

Run research suite directly on a case study, no UI needed (T3, T5, T6, T7):
    python build_tables.py --research-case holoclean_hospital

...or on any CSV:
    python build_tables.py --research-file data/real_world/02_hospital.csv

...or reuse a manifest you already downloaded from the app:
    python build_tables.py --research-manifest selfhealds_research_run.json

Before/after quality for one or more real-world cases, no UI needed (T4):
    python build_tables.py --case-report holoclean_hospital holoclean_flights
    python build_tables.py --case-report all        # all 7 cases with a local CSV

LLM baseline comparison table (T8), from run_llm_baseline.py's output:
    python build_tables.py --llm-baseline research_outputs/llm_baseline_*.json

Combine any of the above in one call; each writes its own section into
--out-dir (default: tables/) as both .md and .csv.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from backend.case_studies import CASE_STUDIES, get_case, load_case_dataset, CACHE  # noqa: E402
from backend.models import ISSUE_TAXONOMY  # noqa: E402
from backend.rules import POLICY, POLICY_RATIONALE  # noqa: E402
from backend.pipeline import heal  # noqa: E402
from backend.research import run_research_suite  # noqa: E402


def _write(df: pd.DataFrame, name: str, out_dir: Path, title: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}.csv"
    md_path = out_dir / f"{name}.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(f"### {title}\n\n" + df.to_markdown(index=False))
    print(f"Wrote {csv_path} and {md_path} ({len(df)} rows).")


# --------------------------------------------------------------------------
# T1 — Case-study characteristics
# --------------------------------------------------------------------------

def build_t1(out_dir: Path) -> None:
    rows = []
    for c in CASE_STUDIES:
        path = CACHE / c.get("local_filename", "")
        n_rows = n_cols = None
        if path.exists():
            df = pd.read_csv(path, low_memory=False)
            n_rows, n_cols = len(df), len(df.columns)
        gt_name = c.get("ground_truth_filename")
        gt_available = "Yes (independent)" if gt_name and (CACHE / gt_name).exists() else \
                        "Yes, not downloaded locally" if gt_name else "No"
        rows.append({
            "Case ID": c["id"],
            "Name": c["name"],
            "Domain": c["domain"],
            "Year": c["year"],
            "Rows x Cols": f"{n_rows} x {n_cols}" if n_rows is not None else "local CSV missing",
            "Ground truth": gt_available,
            "Source": c["organization"],
        })
    _write(pd.DataFrame(rows), "T1_case_study_characteristics", out_dir,
           "T1 — Case-study characteristics")


# --------------------------------------------------------------------------
# T2 — Quality taxonomy / rule catalogue summary
# --------------------------------------------------------------------------

def build_t2(out_dir: Path) -> None:
    by_category: dict[str, list[str]] = {}
    for issue_type, category in ISSUE_TAXONOMY.items():
        by_category.setdefault(category, []).append(issue_type)
    taxonomy_rows = [
        {"Category": cat, "# Issue types": len(types), "Example issue types": ", ".join(sorted(types)[:4]) + ("..." if len(types) > 4 else "")}
        for cat, types in sorted(by_category.items())
    ]
    _write(pd.DataFrame(taxonomy_rows), "T2a_issue_taxonomy_by_category", out_dir,
           "T2a — Issue taxonomy, grouped by quality dimension")

    policy_rows = [
        {"Policy parameter": k, "Value": v, "Rationale": POLICY_RATIONALE.get(
            {"age_upper_bound": "age_upper_bound", "phone_min_digits": "phone_range",
             "phone_max_digits": "phone_range", "missing_max_rate": "missing_rate",
             "missing_min_observed": "missing_min_observed", "outlier_iqr_multiplier": "iqr_multiplier",
             "quality_error_weight": "quality_weights", "quality_warning_weight": "quality_weights",
             "quality_info_weight": "quality_weights"}.get(k, k), "")}
        for k, v in POLICY.items()
    ]
    _write(pd.DataFrame(policy_rows), "T2b_policy_parameters", out_dir,
           "T2b — Repair policy parameters and rationale")


# --------------------------------------------------------------------------
# T3, T5, T6, T7 — from a research suite run (direct or from manifest)
# --------------------------------------------------------------------------

def _emit_research_tables(r: dict, label: str, out_dir: Path) -> None:
    cat = r.get("category_breakdown")
    if cat is not None:
        _write(pd.DataFrame(cat), f"T3_error_category_{label}", out_dir,
               f"T3 — Detection performance by error category ({label})")

    base = r.get("baseline", {})
    base_rows = [
        {"Method": k, "Precision (%)": v.get("precision_pct"), "Recall (%)": v.get("recall_pct"),
         "F1 (%)": v.get("f1_pct"), "Predictions": v.get("predictions")}
        for k, v in base.items() if isinstance(v, dict)
    ]
    if base_rows:
        _write(pd.DataFrame(base_rows), f"T5_baseline_comparison_{label}", out_dir,
               f"T5 — SelfHealDS vs. minimal baseline ({label})")

    sweep_summary = r.get("sweep_summary")
    if sweep_summary is not None:
        _write(pd.DataFrame(sweep_summary), f"T6_corruption_sensitivity_{label}", out_dir,
               f"T6 — Corruption-level sensitivity, mean +/- SD ({label})")

    ablation = r.get("ablation")
    if ablation is not None:
        _write(pd.DataFrame(ablation), f"T7_ablation_{label}", out_dir,
               f"T7 — Component ablation ({label})")


def run_research_direct(df: pd.DataFrame, label: str, out_dir: Path) -> None:
    print(f"Running research suite directly on '{label}' ({len(df)} rows)... "
          "this reuses backend.research.run_research_suite verbatim, no UI needed.")
    from backend.research import to_jsonable
    r = run_research_suite(df)
    _emit_research_tables(r, label, out_dir)
    manifest_path = out_dir / f"research_manifest_{label}.json"
    manifest_path.write_text(json.dumps(to_jsonable(r), indent=2, default=str))
    print(f"Also saved full manifest: {manifest_path} (same content the app's download button gives you).")


def run_research_from_manifest(path: Path, out_dir: Path) -> None:
    r = json.loads(path.read_text())
    _emit_research_tables(r, path.stem, out_dir)


# --------------------------------------------------------------------------
# T4 — Before-vs-after quality metrics per real-world case
# --------------------------------------------------------------------------

def build_t4(case_ids: list[str], out_dir: Path) -> None:
    if case_ids == ["all"]:
        case_ids = [c["id"] for c in CASE_STUDIES]
    rows = []
    for cid in case_ids:
        case = get_case(cid)
        if case is None:
            print(f"  skip: unknown case id '{cid}'")
            continue
        df, meta = load_case_dataset(case)
        if df is None:
            print(f"  skip: {case['name']} — {meta.get('reason')}")
            continue
        print(f"  running heal() on {case['name']} ({len(df)} rows)...")
        result = heal(df)
        bq, aq = result["before_quality"], result["after_quality"]
        val = result["validation"]
        rows.append({
            "Case": case["name"],
            "Rows": len(df),
            "Quality before": bq["score"],
            "Quality after": aq["score"],
            "Errors before": bq["errors"],
            "Errors after": aq["errors"],
            "Repair events": len(result["repair_log"]),
            "Automatic repairs": val["automatic_repairs"],
            "Unsafe repairs": val["unsafe_repairs"],
            "Review rows": result["quarantined_rows"],
            "Validation passed": val["validation_passed"],
        })
    if rows:
        _write(pd.DataFrame(rows), "T4_before_after_case_studies", out_dir,
               "T4 — Before-vs-after quality metrics per real-world case")
    else:
        print("No T4 rows produced — no requested case had a local CSV available.")


# --------------------------------------------------------------------------
# T8 — SelfHealDS vs. LLM baseline
# --------------------------------------------------------------------------

def build_t8(manifest_paths: list[str], out_dir: Path) -> None:
    files: list[Path] = []
    for pattern in manifest_paths:
        files.extend(Path(p) for p in glob.glob(pattern))
    files = sorted(set(files))
    if not files:
        print("No LLM baseline manifest files matched.")
        return
    rows = []
    for f in files:
        m = json.loads(f.read_text())
        se = m.get("selfhealds_evaluation", {})
        llm = m.get("llm_repair_scores", {})
        rows.append({
            "Dataset": m.get("reference_label"),
            "Model": m.get("model"),
            "Rows sent to LLM": m.get("rows_sent_to_llm"),
            "SelfHealDS repair accuracy (%)": se.get("repair_accuracy_pct"),
            "LLM repair accuracy (%)": llm.get("repair_accuracy_pct"),
            "SelfHealDS unsafe-repair rate (%)": se.get("unsafe_repair_rate_pct"),
            "LLM unsafe-repair rate (%)": llm.get("unsafe_repair_rate_pct"),
            "LLM collateral changes": m.get("llm_collateral_changes", {}).get("collateral_change_count"),
            "Domain-trap violations": (m.get("domain_trap_check") or {}).get("incorrectly_altered"),
            "manifest": f.name,
        })
    _write(pd.DataFrame(rows), "T8_llm_baseline_comparison", out_dir,
           "T8 — SelfHealDS vs. LLM baseline")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=Path("tables"))
    ap.add_argument("--t1", action="store_true", help="Build T1 (case-study characteristics).")
    ap.add_argument("--t2", action="store_true", help="Build T2 (issue taxonomy + policy parameters).")
    ap.add_argument("--research-case", help="Case id: run the research suite directly on this case's local CSV (T3, T5, T6, T7).")
    ap.add_argument("--research-file", help="Any CSV path: run the research suite directly on it (T3, T5, T6, T7).")
    ap.add_argument("--research-manifest", help="Path to an already-downloaded selfhealds_research_run.json (T3, T5, T6, T7).")
    ap.add_argument("--case-report", nargs="+", metavar="CASE_ID",
                     help="Real-world case id(s), or 'all', to run heal() directly and build T4.")
    ap.add_argument("--llm-baseline", nargs="+", metavar="PATH_OR_GLOB",
                     help="Path(s)/glob(s) to run_llm_baseline.py JSON manifests, to build T8.")
    args = ap.parse_args()

    did_anything = False

    if args.t1:
        build_t1(args.out_dir); did_anything = True
    if args.t2:
        build_t2(args.out_dir); did_anything = True

    if args.research_case:
        case = get_case(args.research_case)
        if case is None:
            raise SystemExit(f"Unknown case id: {args.research_case}")
        df, meta = load_case_dataset(case)
        if df is None:
            raise SystemExit(f"Local CSV missing for {args.research_case}: {meta.get('reason')}")
        run_research_direct(df, args.research_case, args.out_dir); did_anything = True

    if args.research_file:
        path = Path(args.research_file)
        if not path.exists():
            raise SystemExit(f"Not found: {path}")
        df = pd.read_csv(path, low_memory=False)
        run_research_direct(df, path.stem, args.out_dir); did_anything = True

    if args.research_manifest:
        run_research_from_manifest(Path(args.research_manifest), args.out_dir); did_anything = True

    if args.case_report:
        build_t4(args.case_report, args.out_dir); did_anything = True

    if args.llm_baseline:
        build_t8(args.llm_baseline, args.out_dir); did_anything = True

    if not did_anything:
        print("Nothing to do — pass at least one of --t1 / --t2 / --research-case / "
              "--research-file / --research-manifest / --case-report / --llm-baseline. "
              "See --help for examples.")


if __name__ == "__main__":
    main()
