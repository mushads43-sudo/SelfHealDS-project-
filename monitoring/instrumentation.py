"""Non-invasive instrumentation of the SelfHealDS backend.

`instrument_backend()` replaces three public functions with thin wrappers that
call the ORIGINAL function, return its result unchanged, and record metrics
on the side:

    backend.pipeline.assess              -> stage="assess"
    backend.pipeline.heal                -> stage="heal"
    backend.research.run_research_suite  -> stage="research"

app.py imports these names with `from backend.pipeline import assess, heal`
on every Streamlit rerun, so it automatically receives the wrapped versions.
backend.case_studies.evaluate_case imports them lazily too, so real-world case
studies are measured as well. No file in backend/ or app.py is modified.

Metric recording is wrapped in try/except: a monitoring problem can never
change or break a SelfHealDS result.
"""
from __future__ import annotations

import functools
import logging
import os
import time

from . import metrics as m

log = logging.getLogger("selfhealds.instrumentation")
_MARK = "__selfhealds_instrumented__"


def _safe(fn):
    @functools.wraps(fn)
    def inner(*a, **kw):
        try:
            fn(*a, **kw)
        except Exception:  # pragma: no cover - defensive
            log.exception("metric recording failed")
    return inner


@_safe
def _record_assess(df, result):
    rows, cols = int(df.shape[0]), int(df.shape[1])
    m.ROWS_PROCESSED.labels("assess").inc(rows)
    m.DATASET_ROWS.set(rows)
    m.DATASET_COLUMNS.set(cols)
    for issue in result.get("issues", []):
        m.ISSUES_DETECTED.labels(str(issue.category), str(issue.severity)).inc()
    m.QUALITY_SCORE.labels("before").set(float(result["quality"]["score"]))
    status = (result.get("ml_model") or {}).get("status") or {}
    m.ML_STATUS.labels(str(status.get("status", "unknown"))).inc()
    m.ML_ANOMALIES.set(float(status.get("anomaly_count", 0) or 0))


@_safe
def _record_heal(df, result):
    m.ROWS_PROCESSED.labels("heal").inc(int(df.shape[0]))
    m.QUALITY_SCORE.labels("before").set(float(result["before_quality"]["score"]))
    m.QUALITY_SCORE.labels("after").set(float(result["after_quality"]["score"]))
    for event in result.get("repair_log", []):
        status = str(event.get("status", "unknown"))
        # Collapse free-text statuses into a bounded label set.
        low = status.lower()
        if status.startswith("Automatically"):
            bucket = "automatic"      # "Automatically repaired" / "Automatically imputed"
        elif "quarantin" in low:
            bucket = "quarantined"    # "Quarantined unresolved error"
        elif "review" in low:
            bucket = "review"         # "Retained for review", "Moved to review-required dataset"
        else:
            bucket = "other"
        m.REPAIRS.labels(bucket).inc()
        if event.get("validation_status") in {"PASSED", "FAILED"}:
            m.REPAIR_VALIDATION.labels(event["validation_status"]).inc()
    m.QUARANTINED_ROWS.set(int(result.get("quarantined_rows", 0)))
    v = result.get("validation", {})
    m.ERROR_REDUCTION.set(float(v.get("error_reduction", 0)))
    passed = bool(v.get("validation_passed", False))
    m.VALIDATION_PASSED.set(1 if passed else 0)
    if not passed:
        m.VALIDATION_FAILURES.inc()


@_safe
def _record_research(df, result):
    ev = result.get("evaluation", {})
    for name, key in {
        "precision": "precision", "recall": "recall", "f1": "f1",
        "repair_accuracy": "repair_accuracy_pct", "unsafe_repair_rate": "unsafe_repair_rate_pct",
        "clean_false_positive_rate": "clean_false_positive_rate_pct", "safe_repair_coverage": "safe_repair_coverage_pct",
    }.items():
        val = ev.get(key)
        if isinstance(val, (int, float)) and val == val:  # skip None / NaN
            m.RESEARCH_METRIC.labels(name).set(float(val))
    m.RESEARCH_LATENCY.labels("detection").set(float(result.get("detection_latency_ms", 0)))
    m.RESEARCH_LATENCY.labels("healing").set(float(result.get("healing_latency_ms", 0)))


def _wrap(stage, original, recorder):
    if getattr(original, _MARK, False):
        return original

    @functools.wraps(original)
    def wrapper(df, *args, **kwargs):
        start = time.perf_counter()
        try:
            result = original(df, *args, **kwargs)
        except Exception:
            m.PIPELINE_RUNS.labels(stage, "error").inc()
            m.PIPELINE_DURATION.labels(stage).observe(time.perf_counter() - start)
            raise
        m.PIPELINE_DURATION.labels(stage).observe(time.perf_counter() - start)
        m.PIPELINE_RUNS.labels(stage, "success").inc()
        recorder(df, result)
        return result

    setattr(wrapper, _MARK, True)
    return wrapper


def instrument_backend() -> None:
    """Idempotently wrap the backend entry points and publish build info."""
    from backend import pipeline, research

    pipeline.assess = _wrap("assess", pipeline.assess, _record_assess)
    pipeline.heal = _wrap("heal", pipeline.heal, _record_heal)
    research.run_research_suite = _wrap("research", research.run_research_suite, _record_research)

    from backend.evaluation import EVAL_VERSION
    from backend.ml_anomaly import MODEL_NAME, RANDOM_STATE

    m.BUILD_INFO.info({
        "version": os.getenv("APP_VERSION", "dev"),
        "git_sha": os.getenv("GIT_SHA", "unknown"),
        "ruleset_version": research.RULESET_VERSION,
        "evaluation_version": EVAL_VERSION,
        "ml_model": MODEL_NAME,
        "ml_random_state": str(RANDOM_STATE),
    })


def bootstrap() -> None:
    """Called by streamlit_app.py on every rerun; all steps are idempotent."""
    instrument_backend()
    m.start_metrics_server()
