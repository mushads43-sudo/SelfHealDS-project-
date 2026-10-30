"""Headless UI smoke test using Streamlit's official AppTest harness.

Renders the ORIGINAL app.py and the production entry point without a
browser, and fails on any uncaught exception.
"""
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_app_renders_without_exception():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert any("SelfHealDS" in m.value for m in at.markdown)


def test_production_entrypoint_renders(monkeypatch):
    monkeypatch.setenv("SELFHEALDS_METRICS_ENABLED", "false")  # no port binding in tests
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=60).run()
    assert not at.exception
