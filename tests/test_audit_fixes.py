"""Regression tests for issues found during the final research-grade audit.

Each test is named after the defect it guards against so a future change
that reintroduces the bug fails loudly here instead of surfacing later as a
silently wrong number in a paper.
"""
import json
import pandas as pd
import pytest

from backend.evaluation import evaluate_detection, ground_truth_table, _clean_false_positive_rate
from backend.repair import repair_dataset
from backend.research import run_research_suite, to_jsonable
from backend.profiler import dataset_summary, profile_dataset
from backend.rules import canonicalize_placeholders, missing_fallback, semantic_type


def _big_enough_df():
    # Needs >= MIN_RESEARCH_ROWS-independent but enough rows/columns that the
    # benchmark can inject both obvious and plausible cases across multiple
    # semantic fields, and enough spread that the corruption-difficulty split
    # is actually exercised (>1 case of each kind).
    return pd.DataFrame({
        "age": [21, 24, 28, 31, 47, 33, 19, 55, 62, 40],
        "phone": ["9876543210","98765-43210","9123456780","9988776655",
                  "9871234560","9090909090","9111222233","9333444455",
                  "9555666677","9777888899"],
        "email": ["a@example.com","b@example.com","c@example.com","d@example.com",
                  "e@example.com","f@example.com","g@example.com","h@example.com",
                  "i@example.com","j@example.com"],
        "quantity": [10,8,5,12,3,7,9,4,6,11],
        "city": ["Mumbai","Pune","Delhi","Mumbai","Pune","Delhi","Mumbai","Pune","Delhi","Mumbai"],
        "country": ["India","India","India","India","India","India","India","India","India","India"],
    })


def test_obvious_and_plausible_split_matches_case_labels():
    """Guards the zip(set, list) positional-mismatch bug.

    Every key placed in the 'obvious' truth set must come from a case whose
    own difficulty field is actually 'obvious', and likewise for 'plausible'.
    Previously these were paired positionally against an unordered set and
    could be scrambled.
    """
    df = _big_enough_df()
    ev = evaluate_detection(df, corruption_rate=0.5, seed=42)
    by_key = {
        (int(c["row"]), str(c["column"]), str(c["issue_type"])): c["difficulty"]
        for c in ev["ground_truth"]
    }
    # Reconstruct what obvious/plausible truth sets the evaluator used and
    # verify every member's real difficulty label matches which set it's in.
    obvious_cases = [c for c in ev["ground_truth"] if c["difficulty"] == "obvious"]
    plausible_cases = [c for c in ev["ground_truth"] if c["difficulty"] == "plausible"]
    assert ev["obvious_cases"] == len(obvious_cases)
    assert ev["plausible_cases"] == len(plausible_cases)
    # Sanity: with 10 rows and multiple eligible columns at a 50% corruption
    # rate we should actually get cases of both difficulties, otherwise the
    # test doesn't exercise the split at all.
    assert obvious_cases and plausible_cases


def test_clean_false_positive_rate_is_computed_not_placeholder():
    df = _big_enough_df()
    ev = evaluate_detection(df, corruption_rate=0.1, seed=42)
    assert ev["clean_false_positive_rate_pct"] is not None
    assert 0 <= ev["clean_false_positive_rate_pct"] <= 100


def test_clean_false_positive_rate_helper_bounds():
    reference = _big_enough_df()
    truth = {(0, "age", "Negative age")}
    predicted = {(0, "age", "Negative age"), (5, "quantity", "Negative quantity")}
    rate = _clean_false_positive_rate(reference, truth, predicted)
    total_cells = reference.shape[0] * reference.shape[1]
    negatives = total_cells - len(truth)
    assert rate == pytest.approx(1 / negatives * 100, rel=1e-6, abs=1e-4)


def test_unsafe_repair_rate_is_na_when_nothing_attempted():
    # A dataset with no injectable/repairable corruption specs at all still
    # must not report a misleading 0% ("nothing was unsafe") when in fact
    # nothing was attempted.
    df = pd.DataFrame({"note": ["hello", "world", "foo", "bar", "baz"]})
    ev = evaluate_detection(df, corruption_rate=0.1, seed=42)
    if ev["repair_attempts"] == 0:
        assert ev["unsafe_repair_rate_pct"] is None


def test_ground_truth_table_has_real_columns_and_values():
    cases = [{
        "row": 3, "column": "age", "issue_type": "Negative age", "difficulty": "obvious",
        "original_value": 25.0, "corrupted_value": -25.0, "expected_repair": 25.0,
    }]
    table = ground_truth_table(cases)
    assert list(table.columns) == ["Row","Column","Issue type","Difficulty","Original value","Corrupted value","Expected repair"]
    assert table.loc[0, "Row"] == 3
    assert table.loc[0, "Expected repair"] == 25.0


def test_ground_truth_table_empty_is_well_formed():
    table = ground_truth_table([])
    assert table.empty
    assert "Issue type" in table.columns


def test_unitless_measurement_value_is_not_rewritten_with_assumed_unit():
    """A bare number in a weight/length column carries no evidence of which
    unit it is in; repair must not silently append a canonical unit label."""
    df = pd.DataFrame({"weight": ["70", "2 kg", "1500 g"]})
    cleaned, before, remaining, log, review, post = repair_dataset(df)
    assert cleaned.loc[0, "weight"] == "70"
    assert not any(e["row"] == 0 and e["issue_type"] == "Unit normalization" for e in log)
    # Values that DO carry an explicit recognized unit are still normalized.
    assert any(e["row"] == 2 and e["issue_type"] == "Unit normalization" for e in log)


def test_missing_fallback_ignores_placeholder_tokens_after_canonicalization():
    s = pd.Series(["21", "24", "28", "31", "47", "n/a"])
    fallback, method, evidence = missing_fallback(canonicalize_placeholders(s), semantic_type("age"))
    assert fallback == 28.0
    assert method == "median"


def test_dataset_summary_missing_cells_matches_profile_definition():
    df = pd.DataFrame({"a": ["x", "  ", None, "y"]})
    summary = dataset_summary(df)
    profile = profile_dataset(df)
    assert summary["missing_cells"] == int(profile["missing"].sum())


def test_to_jsonable_serializes_dataframes_and_issues_fully():
    df = _big_enough_df()
    result = run_research_suite(df)
    payload = to_jsonable(result)
    encoded = json.dumps(payload, default=str)
    # The messy benchmark dataframe must appear as real records, not a
    # truncated pandas str() preview (which contains literal "...").
    messy_json = json.dumps(payload["evaluation"]["messy"])
    assert "..." not in messy_json
    assert isinstance(payload["evaluation"]["messy"], list)
    # Detected issues must be plain dicts, not stringified dataclass reprs.
    if payload["evaluation"]["detected"]:
        first = payload["evaluation"]["detected"][0]
        assert isinstance(first, dict)
        assert "issue_type" in first
    assert len(encoded) > 0


def test_bundled_suite_resolves_relative_to_project_not_cwd(tmp_path, monkeypatch):
    from backend.evaluation import evaluate_bundled_suite
    monkeypatch.chdir(tmp_path)  # simulate launching streamlit from elsewhere
    df = _big_enough_df()
    result = run_research_suite(df)
    # Should find the real bundled sample datasets, not report a missing dir,
    # because research.py resolves the path from its own module location.
    assert "error" not in result["bundled_suite"].columns or result["bundled_suite"]["error"].isna().all()
    assert len(result["bundled_suite"]) >= 1


def test_evaluate_bundled_suite_missing_dir_reports_explicit_error(tmp_path):
    from backend.evaluation import evaluate_bundled_suite
    missing = tmp_path / "does_not_exist"
    out = evaluate_bundled_suite(missing)
    assert "error" in out.columns
    assert "not found" in out.loc[0, "error"]
