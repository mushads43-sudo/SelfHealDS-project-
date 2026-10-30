"""Prometheus metric definitions for SelfHealDS.

Metric families map onto the four monitoring goals:

* Application performance - pipeline latency, run counts, failures, UI reruns.
* Model / research metrics - quality score, repairs, quarantine, validation
  gate, Isolation Forest anomalies, benchmark precision/recall/F1.
* Service health      - build info, up-time (process_start_time_seconds).
* System resources    - process CPU/memory from prometheus_client's default
  collectors; node/pod resources come from node-exporter/cAdvisor in K8s.

Module-level objects are created once per Python process. Streamlit re-runs
the page script on every interaction but keeps imported modules cached in
sys.modules, so these collectors are never registered twice.
"""
from __future__ import annotations

import logging
import os
import threading

from prometheus_client import Counter, Gauge, Histogram, Info, start_http_server

log = logging.getLogger("selfhealds.metrics")

# Latency buckets sized for SelfHealDS: small CSVs assess in <100 ms, the
# research suite on ~1k rows takes tens of seconds.
_PIPELINE_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300)
_UI_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)

BUILD_INFO = Info("selfhealds_build", "SelfHealDS build and ruleset metadata")

PIPELINE_RUNS = Counter(
    "selfhealds_pipeline_runs_total",
    "Pipeline invocations by stage and outcome",
    ["stage", "status"],
)
PIPELINE_DURATION = Histogram(
    "selfhealds_pipeline_duration_seconds",
    "Wall-clock duration of pipeline stages",
    ["stage"],
    buckets=_PIPELINE_BUCKETS,
)
ROWS_PROCESSED = Counter("selfhealds_rows_processed_total", "Rows passed through a pipeline stage", ["stage"])
DATASET_ROWS = Gauge("selfhealds_last_dataset_rows", "Rows in the most recently assessed dataset")
DATASET_COLUMNS = Gauge("selfhealds_last_dataset_columns", "Columns in the most recently assessed dataset")

ISSUES_DETECTED = Counter(
    "selfhealds_issues_detected_total",
    "Findings raised by the detector, by taxonomy category and severity",
    ["category", "severity"],
)
QUALITY_SCORE = Gauge(
    "selfhealds_quality_score",
    "Severity-weighted quality indicator (0-100) of the last run",
    ["phase"],  # before | after
)
REPAIRS = Counter("selfhealds_repairs_total", "Repair-log events by status", ["status"])
REPAIR_VALIDATION = Counter(
    "selfhealds_repair_validation_total",
    "Automatic repairs checked by the post-repair validation gate",
    ["result"],  # PASSED | FAILED
)
QUARANTINED_ROWS = Gauge("selfhealds_last_quarantined_rows", "Rows sent to review/quarantine by the last heal run")
ERROR_REDUCTION = Gauge("selfhealds_last_error_reduction", "Error-level findings removed by the last heal run")
VALIDATION_PASSED = Gauge("selfhealds_last_validation_passed", "1 if the last heal run passed the validation gate")
VALIDATION_FAILURES = Counter("selfhealds_validation_gate_failures_total", "Heal runs whose validation gate did not pass")

ML_ANOMALIES = Gauge("selfhealds_ml_anomalies_last", "Isolation Forest anomaly rows in the last assessed dataset")
ML_STATUS = Counter("selfhealds_ml_model_runs_total", "Isolation Forest runs by status", ["status"])

RESEARCH_METRIC = Gauge(
    "selfhealds_research_metric",
    "Latest Research Evaluation metric (percent) from the controlled-corruption benchmark",
    ["metric"],
)
RESEARCH_LATENCY = Gauge("selfhealds_research_latency_ms", "Latency reported by the last research suite", ["phase"])

UI_RERUNS = Counter("selfhealds_ui_reruns_total", "Streamlit script reruns by sidebar page", ["page"])
UI_RERUN_DURATION = Histogram(
    "selfhealds_ui_rerun_duration_seconds", "Server-side render time of one Streamlit rerun", buckets=_UI_BUCKETS
)

_server_lock = threading.Lock()
_server_started = False


def start_metrics_server() -> bool:
    """Start the /metrics HTTP endpoint once per process.

    Controlled by environment variables:
      SELFHEALDS_METRICS_ENABLED (default "true")
      SELFHEALDS_METRICS_PORT    (default 8000)
    Returns True when the endpoint is (already) running.
    """
    global _server_started
    if os.getenv("SELFHEALDS_METRICS_ENABLED", "true").lower() not in {"1", "true", "yes"}:
        return False
    with _server_lock:
        if _server_started:
            return True
        port = int(os.getenv("SELFHEALDS_METRICS_PORT", "8000"))
        try:
            start_http_server(port, addr="0.0.0.0")
            _server_started = True
            log.info("Prometheus metrics endpoint listening on :%s/metrics", port)
        except OSError as exc:  # port already bound (e.g. second Streamlit session in dev)
            log.warning("Metrics endpoint not started on port %s: %s", port, exc)
            _server_started = True
        return _server_started
