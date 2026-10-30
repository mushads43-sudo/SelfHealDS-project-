"""SelfHealDS model-quality gate for CI/CD.

Re-uses the project's OWN research protocol (backend.evaluation.
evaluate_detection - controlled corruption with ground truth) and fails the
pipeline when detection / repair quality regresses below mlops/thresholds.toml.

Usage:
    python mlops/validate_model.py                    # human-readable + exit code
    python mlops/validate_model.py --report out.json  # also write JSON report

Exit code 0 = gate passed, 1 = gate failed.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from backend.evaluation import EVAL_VERSION, evaluate_detection  # noqa: E402
from backend.pipeline import assess, heal  # noqa: E402
from backend.profiler import dataset_fingerprint  # noqa: E402
from backend.research import RULESET_VERSION  # noqa: E402


def _mean(values):
    vals = [v for v in values if isinstance(v, (int, float)) and v == v]
    return round(statistics.mean(vals), 2) if vals else None


def run_benchmark(cfg):
    path = ROOT / cfg["dataset"]
    df = pd.read_csv(path, low_memory=False)
    runs = []
    for seed in cfg["seeds"]:
        t = time.perf_counter()
        ev = evaluate_detection(df, corruption_rate=cfg["corruption_rate"], seed=seed)
        runs.append({
            "seed": seed,
            "cases": ev["actual_corruption_cases"],
            "precision": ev["precision"], "recall": ev["recall"], "f1": ev["f1"],
            "repair_accuracy": ev["repair_accuracy_pct"],
            "unsafe_repair_rate": ev["unsafe_repair_rate_pct"] or 0.0,
            "clean_fpr": ev["clean_false_positive_rate_pct"] or 0.0,
            "seconds": round(time.perf_counter() - t, 2),
        })
    summary = {k: _mean(r[k] for r in runs) for k in ("precision", "recall", "f1", "repair_accuracy", "clean_fpr", "seconds")}
    summary["unsafe_repair_rate"] = max(r["unsafe_repair_rate"] for r in runs)  # worst case, not mean
    return {"dataset": cfg["dataset"], "fingerprint": dataset_fingerprint(df), "rows": len(df), "runs": runs, "summary": summary}


def run_smoke(sample_dir):
    results = []
    for path in sorted((ROOT / sample_dir).glob("*.csv")):
        df = pd.read_csv(path)
        a = assess(df)
        h = heal(df)
        results.append({
            "dataset": path.name, "rows": len(df), "findings": len(a["issues"]),
            "quality_before": h["before_quality"]["score"], "quality_after": h["after_quality"]["score"],
            "failed_repairs": h["validation"]["unsafe_repairs"],
        })
    return results


def check(summary, smoke, t, baseline_failed):
    checks = [
        ("precision", summary["precision"], ">=", t["min_precision_pct"]),
        ("recall", summary["recall"], ">=", t["min_recall_pct"]),
        ("f1", summary["f1"], ">=", t["min_f1_pct"]),
        ("repair_accuracy", summary["repair_accuracy"], ">=", t["min_repair_accuracy_pct"]),
        ("unsafe_repair_rate (worst seed)", summary["unsafe_repair_rate"], "<=", t["max_unsafe_repair_rate_pct"]),
        ("clean_false_positive_rate", summary["clean_fpr"], "<=", t["max_clean_false_positive_rate_pct"]),
        ("mean_runtime_seconds", summary["seconds"], "<=", t["max_mean_runtime_seconds"]),
        ("samples with new validation-gate failures",
         sum(1 for s in smoke if s["failed_repairs"] > baseline_failed.get(s["dataset"], 0)), "<=", 0),
    ]
    out = []
    for name, value, op, limit in checks:
        ok = value is not None and (value >= limit if op == ">=" else value <= limit)
        out.append({"check": name, "value": value, "rule": f"{op} {limit}", "passed": bool(ok)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "mlops" / "thresholds.toml"))
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    cfg = tomllib.loads(Path(args.config).read_text())

    bench = run_benchmark(cfg["benchmark"])
    smoke = run_smoke(cfg["smoke"]["sample_dir"])
    checks = check(bench["summary"], smoke, cfg["thresholds"], cfg["smoke"].get("baseline_failed_repairs", {}))
    passed = all(c["passed"] for c in checks)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": os.getenv("GITHUB_SHA", "local"),
        "ruleset_version": RULESET_VERSION, "evaluation_version": EVAL_VERSION,
        "benchmark": bench, "smoke": smoke, "checks": checks, "passed": passed,
    }

    print(f"SelfHealDS quality gate | ruleset {RULESET_VERSION} | eval {EVAL_VERSION}")
    print(f"Benchmark: {bench['dataset']} ({bench['rows']} rows, fingerprint {bench['fingerprint']})")
    for r in bench["runs"]:
        print(f"  seed {r['seed']}: P={r['precision']} R={r['recall']} F1={r['f1']} RA={r['repair_accuracy']} unsafe={r['unsafe_repair_rate']} ({r['seconds']}s)")
    print("Checks:")
    for c in checks:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['check']:<42} {str(c['value']):>8}  {c['rule']}")
    print("QUALITY GATE PASSED" if passed else "QUALITY GATE FAILED")

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2, default=str))
    step_summary = os.getenv("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a") as fh:
            fh.write(f"## SelfHealDS quality gate: {'PASSED' if passed else 'FAILED'}\n\n| Check | Value | Rule | Result |\n|---|---|---|---|\n")
            for c in checks:
                fh.write(f"| {c['check']} | {c['value']} | {c['rule']} | {'pass' if c['passed'] else 'FAIL'} |\n")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
