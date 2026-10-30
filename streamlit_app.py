"""Production entry point for SelfHealDS.

    streamlit run streamlit_app.py      # production / Docker / Kubernetes
    streamlit run app.py                # original research UI, unchanged

This file adds Prometheus monitoring around the ORIGINAL app.py without
editing it: it instruments the backend, starts the /metrics endpoint once per
process, then executes app.py exactly as Streamlit would. Streamlit control-
flow exceptions (st.rerun / st.stop) propagate untouched.
"""
import runpy
import time
from pathlib import Path

import streamlit as st

from monitoring.instrumentation import bootstrap
from monitoring import metrics

bootstrap()

_started = time.perf_counter()
try:
    runpy.run_path(str(Path(__file__).resolve().parent / "app.py"), run_name="__main__")
finally:
    metrics.UI_RERUN_DURATION.observe(time.perf_counter() - _started)
    try:
        metrics.UI_RERUNS.labels(str(st.session_state.get("active_page", "unknown"))).inc()
    except Exception:
        pass
