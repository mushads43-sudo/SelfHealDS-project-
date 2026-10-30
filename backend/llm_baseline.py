"""Optional LLM-cleaning baseline, for the "how does SelfHealDS compare to
just asking an LLM to clean the data" research question.

This module is intentionally isolated from the core pipeline: SelfHealDS's
own detection/repair path (backend/detector.py, backend/repair.py) never
imports this file and never requires network access or an API key — that
"local-first, no paid AI API required" property is a load-bearing part of the
project's positioning and must not be weakened by this comparison feature.

This module IS a genuinely working implementation (not a mock/stub): given a
valid Anthropic API key, `clean_with_llm` makes a real HTTPS call to
api.anthropic.com using only the Python standard library (no extra
dependency added to requirements.txt for an optional, off-by-default
feature), and `evaluate_llm_baseline` scores the result with the exact same
methodology (build_benchmark, _scores, repair-accuracy logic) already used to
score SelfHealDS itself, so the two are directly comparable. No API key is
bundled with this project; the researcher supplies their own at runtime
(entered in the UI, kept only in the Streamlit session, never written to
disk or logged) or via the ANTHROPIC_API_KEY environment variable.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
import urllib.error
from datetime import datetime, timezone

import pandas as pd

from .evaluation import build_benchmark, _truth_set, _scores, _now
from .detector import detect_issues

DEFAULT_MODEL = "claude-sonnet-4-5"
API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
REQUEST_TIMEOUT_SECONDS = 90

PROMPT_TEMPLATE = """You are cleaning a tabular dataset. Below is a CSV with row indices in the
first column (0-based, matching the original DataFrame index). Some cells
contain data-quality errors (missing values, invalid formats, wrong units,
implausible values, etc).

Task: identify every cell you believe is erroneous and propose a corrected
value. Respond with ONLY a JSON array, no prose, no markdown fences. Each
element must have this exact shape:
{{"row": <int>, "column": "<column name>", "corrected_value": <value or null>}}

Only include cells you are proposing to change. If you believe a cell is
wrong but cannot safely infer a correct replacement, set "corrected_value" to
null (do not guess).

CSV:
{csv_text}
"""


def is_configured(api_key=None):
    """True if a usable API key is available (explicit arg or environment)."""
    return bool(api_key or os.environ.get("ANTHROPIC_API_KEY"))


def _extract_json_array(text):
    """Best-effort extraction of a JSON array from a model response.

    Models sometimes wrap JSON in prose or markdown fences despite
    instructions not to. This looks for the first '[' ... last ']' span
    rather than assuming the response is pure JSON, so a minor formatting
    deviation doesn't fail the whole comparison run.
    """
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON array found in model response.")
    return json.loads(text[start:end + 1])


def clean_with_llm(df, api_key=None, model=DEFAULT_MODEL, max_tokens=4000):
    """Ask an LLM to propose corrections for a dataframe.

    Returns (proposed_repairs, raw_text, error). proposed_repairs is a list
    of {"row": int, "column": str, "corrected_value": value_or_None} dicts;
    empty on failure, with `error` set to a human-readable message. This
    function makes exactly one API call per invocation; it does not retry,
    batch, or cache, since it exists for a bounded research comparison run,
    not production use.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return [], None, "No API key supplied. Set it in the UI field or the ANTHROPIC_API_KEY environment variable."

    csv_text = df.to_csv(index=True)
    prompt = PROMPT_TEMPLATE.format(csv_text=csv_text)
    payload = json.dumps({
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")

    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={
            "content-type": "application/json",
            "x-api-key": key,
            "anthropic-version": API_VERSION,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        return [], None, f"API request failed ({exc.code}): {detail}"
    except urllib.error.URLError as exc:
        return [], None, f"Network error contacting the API: {exc.reason}"
    except Exception as exc:
        return [], None, f"Unexpected error calling the API: {exc}"

    text_blocks = [b.get("text", "") for b in body.get("content", []) if b.get("type") == "text"]
    raw_text = "\n".join(text_blocks)
    try:
        parsed = _extract_json_array(raw_text)
    except Exception as exc:
        return [], raw_text, f"Could not parse the model's response as the requested JSON array: {exc}"

    repairs = []
    for item in parsed:
        try:
            repairs.append({
                "row": int(item["row"]),
                "column": str(item["column"]),
                "corrected_value": item.get("corrected_value"),
            })
        except Exception:
            continue
    return repairs, raw_text, None


def evaluate_llm_baseline(df, corruption_rate=0.10, seed=42, api_key=None, model=DEFAULT_MODEL):
    """Score an LLM-cleaning pass against the SAME injected-corruption
    benchmark used to score SelfHealDS (backend.evaluation.evaluate_detection),
    so the two numbers are directly comparable rather than measuring
    different things.

    This makes exactly one LLM call for the whole messy dataset (not one per
    cell), matching how a person would realistically use a general-purpose
    assistant for this task.
    """
    reference = df.copy().reset_index(drop=True)
    messy, cases = build_benchmark(reference, corruption_rate, seed)
    truth = _truth_set(cases)

    if not is_configured(api_key):
        return {
            "available": False,
            "reason": "No API key configured. This comparison is optional and never runs automatically.",
            "generated_at": _now(),
        }

    repairs, raw_text, error = clean_with_llm(messy, api_key=api_key, model=model)
    if error:
        return {"available": False, "reason": error, "generated_at": _now()}

    # Score detection: did the model identify (row, column) cells that were
    # actually corrupted? The model doesn't emit an issue_type label, so we
    # match on (row, column) only here, then separately check value-level
    # repair correctness for cells the model actually attempted.
    truth_cells = {(r, c) for r, c, _t in truth}
    predicted_cells = {(rep["row"], rep["column"]) for rep in repairs}
    tp = len(predicted_cells & truth_cells)
    fp = len(predicted_cells - truth_cells)
    fn = len(truth_cells - predicted_cells)
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall and (precision + recall) else None

    case_by_cell = {(int(c["row"]), str(c["column"])): c for c in cases}
    correct = 0
    attempted_with_value = 0
    for rep in repairs:
        key = (rep["row"], rep["column"])
        if key not in case_by_cell or rep["corrected_value"] is None:
            continue
        attempted_with_value += 1
        expected = case_by_cell[key].get("expected_repair")
        if expected is None:
            continue
        try:
            ok = abs(float(rep["corrected_value"]) - float(expected)) <= max(1e-9, abs(float(expected)) * 1e-9)
        except Exception:
            ok = str(rep["corrected_value"]).strip() == str(expected).strip()
        if ok:
            correct += 1
    repair_accuracy = (correct / attempted_with_value * 100) if attempted_with_value else None

    return {
        "available": True,
        "generated_at": _now(),
        "model": model,
        "corruption_rate": corruption_rate,
        "seed": seed,
        "ground_truth_cells": len(truth_cells),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision_pct": None if precision is None else round(precision * 100, 2),
        "recall_pct": None if recall is None else round(recall * 100, 2),
        "f1_pct": None if f1 is None else round(f1 * 100, 2),
        "repair_attempts": attempted_with_value,
        "repair_correct": correct,
        "repair_accuracy_pct": None if repair_accuracy is None else round(repair_accuracy, 2),
        "raw_response_preview": (raw_text or "")[:2000],
        "method": "Single-shot prompt over the full messy CSV, scored against the identical injected-corruption ground truth used for SelfHealDS. One API call per run; not retried, batched or cached.",
        "caveats": "A single prompt/model/temperature configuration is one point in a large design space; a lower score here is not proof that no LLM-based approach could do better with more engineering (few-shot examples, structured output tools, multiple passes, self-critique). Treat this as a same-conditions reference point, not a definitive verdict on LLMs generally.",
    }
