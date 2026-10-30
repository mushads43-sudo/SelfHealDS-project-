"""Tests for the MLOps monitoring layer.

Guarantees: (1) instrumentation never changes SelfHealDS results,
(2) the expected Prometheus series are produced, (3) wrapping is idempotent.
"""
import pandas as pd
from prometheus_client import REGISTRY, generate_latest

from monitoring.instrumentation import instrument_backend


def _df():
    return pd.DataFrame({
        "age": [-25, 30, 41, None, 28, 35],
        "phone": [" 98765-43210 ", "123", "9876543210", "9123456780", "9988776655", "9876501234"],
        "city": ["Mumbai", "Pune", "Delhi", "Mumbai", "Pune", "Delhi"],
        "country": ["France", "India", "India", "India", "India", "India"],
    })


def test_instrumentation_preserves_results():
    from backend import pipeline
    original_assess, original_heal = pipeline.assess, pipeline.heal
    raw_a, raw_h = original_assess(_df()), original_heal(_df())
    instrument_backend()
    a, h = pipeline.assess(_df()), pipeline.heal(_df())
    # Compare as DataFrames so NaN cells compare equal.
    assert pd.DataFrame(a["issue_table"]).equals(pd.DataFrame(raw_a["issue_table"]))
    assert a["quality"] == raw_a["quality"]
    assert h["cleaned"].equals(raw_h["cleaned"])
    assert h["validation"] == raw_h["validation"]


def test_instrumentation_is_idempotent():
    from backend import pipeline
    instrument_backend()
    first = pipeline.assess
    instrument_backend()
    assert pipeline.assess is first


def test_expected_metrics_are_exported():
    from backend import pipeline
    instrument_backend()
    pipeline.assess(_df())
    pipeline.heal(_df())
    text = generate_latest(REGISTRY).decode()
    for name in (
        "selfhealds_pipeline_runs_total",
        "selfhealds_pipeline_duration_seconds_bucket",
        "selfhealds_issues_detected_total",
        "selfhealds_quality_score",
        "selfhealds_repairs_total",
        "selfhealds_last_validation_passed",
        "selfhealds_build_info",
    ):
        assert name in text, name
    assert REGISTRY.get_sample_value("selfhealds_pipeline_runs_total", {"stage": "heal", "status": "success"}) >= 1
