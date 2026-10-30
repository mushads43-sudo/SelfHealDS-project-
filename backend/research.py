
from datetime import datetime, timezone
from dataclasses import is_dataclass, asdict
from pathlib import Path
import platform
import time
import pandas as pd
from .evaluation import (
    evaluate_detection, compare_with_baseline, run_corruption_sweep,
    summarize_sweep, run_ablation, evaluate_bundled_suite, DEFAULT_RATES, DEFAULT_SEEDS, MIN_RESEARCH_ROWS
)
from .detector import detect_issues
from .repair import repair_dataset
from .profiler import dataset_fingerprint
from .rules import POLICY, POLICY_RATIONALE
from .ml_anomaly import MODEL_NAME, MODEL_VERSION, RANDOM_STATE

RULESET_VERSION = "2026.08-research-v2"

def _now():
    return datetime.now(timezone.utc).isoformat()

def to_jsonable(obj):
    """Recursively convert a research-suite payload into plain JSON-safe types.

    Without this, exporting a run via json.dumps(..., default=str) silently
    turned embedded pandas DataFrames (e.g. the benchmark's messy dataset)
    into a truncated, human-oriented str(df) preview (columns collapsed to
    "...", rows cut off) and turned lists of Issue dataclass instances into
    per-item repr strings. Both defeat the point of a "reproducible research
    manifest": a reviewer opening the JSON should get structured data, not a
    lossy text dump. This function is used at every JSON export point instead
    of relying on json.dumps' default=str fallback for these types.
    """
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    return obj

def _category_breakdown(ev):
    cases=ev.get("ground_truth",[])
    detected=set(tuple(x) for x in ev.get("new_detected",[]))
    groups={}
    for c in cases:
        key=(int(c["row"]),str(c["column"]),str(c["issue_type"]))
        groups.setdefault((c["issue_type"],c.get("difficulty","unknown")),set()).add(key)
    rows=[]
    for (issue_type,difficulty),truth in groups.items():
        pred=detected & truth
        tp=len(pred); fn=len(truth-pred)
        p=tp/(tp+fn) if tp+fn else None
        rows.append({"Issue type":issue_type,"Difficulty":difficulty,"Injected":len(truth),"Detected":len(pred),"Recall (%)":None if p is None else round(p*100,2)})
    return pd.DataFrame(rows)

def run_research_suite(df):
    started=time.perf_counter()
    fp=dataset_fingerprint(df)
    base=detect_issues(df)
    detection_ms=(time.perf_counter()-started)*1000

    heal_start=time.perf_counter()
    cleaned,before,after,log,review,post=repair_dataset(df)
    healing_ms=(time.perf_counter()-heal_start)*1000

    ev=evaluate_detection(df)
    baseline=compare_with_baseline(df)
    sweep=run_corruption_sweep(df)
    sweep_summary=summarize_sweep(sweep)
    ablation=run_ablation(df)
    category=_category_breakdown(ev)
    # Resolve relative to this module's location (project root, one level up
    # from backend/), not the process working directory. Streamlit can be
    # launched from a different cwd, which previously made this silently
    # return an empty, unexplained table instead of the bundled comparison.
    bundled_dir = Path(__file__).resolve().parent.parent / "sample messy datasets"
    bundled=evaluate_bundled_suite(bundled_dir)

    rows=max(1,len(df))
    return {
        "generated_at":_now(),
        "fingerprint":fp,
        "ruleset_version":RULESET_VERSION,
        "environment":{"python":platform.python_version(),"platform":platform.platform()},
        "detection_latency_ms":round(detection_ms,3),
        "healing_latency_ms":round(healing_ms,3),
        "rows_per_second":round(rows/max(healing_ms/1000,1e-9),2),
        "raw_findings":len(base),
        "evaluation":ev,
        "baseline":baseline,
        "sweep":sweep,
        "sweep_summary":sweep_summary,
        "ablation":ablation,
        "category_breakdown":category,
        "bundled_suite":bundled,
        "cleaned_rows":len(cleaned),
        "review_rows":len(review),
        "repair_events":len(log),
        "post_repair_findings":len(post),
        "experimental_protocol":{
            "corruption_rates_pct":[r*100 for r in DEFAULT_RATES],
            "trials_per_rate":len(DEFAULT_SEEDS),
            "seeds":list(DEFAULT_SEEDS),
            "reason_for_rates":"Five, ten, twenty and thirty percent scenarios represent increasing controlled corruption severity; the values are experimental design levels, not claims of real-world prevalence.",
            "reason_for_trials":"Five fixed seeds provide repeated deterministic row selections so reported means and sample standard deviations expose run-to-run variation. They are not a population confidence interval.",
            "baseline":"Minimal missing-value and exact-duplicate detector. It is a transparent internal baseline, not a claim to reproduce a commercial product.",
            "ground_truth":"Each injected case stores row, column, issue type, original value, corrupted value and expected repair when an exact target exists.",
            "minimum_recommended_rows":MIN_RESEARCH_ROWS,
            "small_dataset_reason":f"Datasets below {MIN_RESEARCH_ROWS} rows can make percentage-based corruption levels too coarse for publication-grade benchmarking; use larger independently curated datasets for final paper results.",
        },
        "policy":POLICY,
        "policy_rationale":POLICY_RATIONALE,
        "ml_model":{"name":MODEL_NAME,"version":MODEL_VERSION,"random_state":RANDOM_STATE},
        "metric_validity":"All metrics are snapshots valid only for this dataset fingerprint, ruleset version, evaluation protocol, seeds and execution timestamp."
    }
