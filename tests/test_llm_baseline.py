import json
from unittest.mock import patch
import pandas as pd
from backend import llm_baseline as lb


def _fake_response(payload_obj):
    class _Resp:
        def read(self):
            return json.dumps({"content": [{"type": "text", "text": json.dumps(payload_obj)}]}).encode("utf-8")
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
    return _Resp()


def test_not_configured_without_key():
    assert lb.is_configured() is False
    assert lb.is_configured("sk-x") is True


def test_evaluate_llm_baseline_unconfigured_is_graceful():
    df = pd.DataFrame({"age": [20, 30, 40, 50, 60, 70, 25, 35], "phone": ["9876543210"] * 8})
    r = lb.evaluate_llm_baseline(df)
    assert r["available"] is False
    assert "reason" in r


def test_evaluate_llm_baseline_scores_a_mocked_response():
    df = pd.DataFrame({
        "age": [20, 30, 40, 50, 60, 70, 25, 35, 45, 55],
        "phone": ["9876543210"] * 10,
    })

    # Patch build_benchmark (via the evaluation module it's imported from) so
    # the injected corruption is deterministic and known to this test, and
    # patch the network call so no real API key or network access is needed.
    fake_cases = [{"row": 0, "column": "age", "issue_type": "Negative age",
                   "difficulty": "obvious", "original_value": 20,
                   "corrupted_value": -20, "expected_repair": 20.0}]

    def fake_build_benchmark(reference, rate, seed):
        messy = reference.copy()
        messy.at[0, "age"] = -20
        return messy, fake_cases

    fake_llm_output = [{"row": 0, "column": "age", "corrected_value": 20}]

    with patch.object(lb, "build_benchmark", side_effect=fake_build_benchmark), \
         patch("urllib.request.urlopen", return_value=_fake_response(fake_llm_output)):
        r = lb.evaluate_llm_baseline(df, api_key="sk-fake-test-key")

    assert r["available"] is True
    assert r["true_positives"] == 1
    assert r["false_negatives"] == 0
    assert r["precision_pct"] == 100.0
    assert r["recall_pct"] == 100.0
    assert r["repair_accuracy_pct"] == 100.0


def test_clean_with_llm_handles_malformed_response_gracefully():
    with patch("urllib.request.urlopen", return_value=_fake_response({"not": "a list"})):
        repairs, raw_text, error = lb.clean_with_llm(pd.DataFrame({"a": [1]}), api_key="sk-fake")
    # A JSON object (not array) still round-trips through _extract_json_array's
    # bracket search failing gracefully rather than crashing the caller.
    assert repairs == []
    assert error is not None
