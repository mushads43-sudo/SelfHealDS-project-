import pandas as pd
from backend.pipeline import assess, heal
from backend.evaluation import evaluate_detection

def test_obvious_errors_are_detected():
    df=pd.DataFrame({"age":[-25,30],"phone":[" 123 ","9876543210"],"email":["bad","a@b.com"],"city":["Mumbai","Pune"],"country":["France","India"],"quantity":[-5,2]})
    result=assess(df); types={i.issue_type for i in result["issues"]}
    assert {"Negative age","Invalid phone length","Malformed email","City-country semantic inconsistency","Negative quantity"}.issubset(types)

def test_invalid_phone_is_not_exported_as_clean():
    df=pd.DataFrame({"phone":["123","9876543210"],"age":[25,30]})
    h=heal(df)
    assert len(h["review_required"])>=1
    assert "123" not in h["cleaned"]["phone"].astype(str).tolist()

def test_metrics_are_bounded():
    df=pd.DataFrame({"age":[20,30,40,50],"phone":["9876543210"]*4,"city":["Mumbai"]*4,"country":["India"]*4})
    e=evaluate_detection(df)
    for k in ["precision","recall","f1","repair_accuracy_pct","clean_false_positive_rate_pct"]:
        assert e[k] is None or 0<=e[k]<=100


def test_missing_value_repair_records_derivation():
    df=pd.DataFrame({"age":[21,24,28,31,47,None]})
    h=heal(df)
    events=[x for x in h["repair_log"] if x["issue_type"]=="Missing value"]
    assert events
    event=events[0]
    assert event["repaired_value"]==28.0
    assert "median" in event["reason"].lower()
    assert "16.67%" in event["reason"]
    assert "5 parseable" in event["reason"].lower()

def test_insufficient_missing_age_is_not_fabricated():
    df=pd.DataFrame({"age":[21,None,28]})
    h=heal(df)
    events=[x for x in h["repair_log"] if x["issue_type"]=="Missing value"]
    assert not events
    assert len(h["review_required"]) >= 1

def test_invalid_phone_is_quarantined_not_invented():
    df=pd.DataFrame({"phone":["123","9876543210"],"age":[25,30]})
    h=heal(df)
    assert "123" not in h["cleaned"]["phone"].astype(str).tolist()
    assert any("excluded from the validated clean output" in x["reason"].lower() for x in h["repair_log"])

def test_policy_parameters_are_exposed():
    from backend.rules import POLICY
    assert POLICY["missing_max_rate"] == 0.20
    assert POLICY["missing_min_observed"] == 5
    assert POLICY["phone_min_digits"] == 7
    assert POLICY["phone_max_digits"] == 15
