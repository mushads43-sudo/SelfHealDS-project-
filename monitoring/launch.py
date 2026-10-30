"""Container entry point: `python -m monitoring.launch [streamlit args]`.

Starts the Prometheus /metrics endpoint and instruments the backend BEFORE
Streamlit starts, so the scrape target is up even when no browser session is
open. Streamlit then runs in the same Python process, so the app and the
metrics endpoint share one Prometheus registry.
"""
import sys

from streamlit.web import cli as stcli

from monitoring.instrumentation import bootstrap


def main() -> None:
    bootstrap()
    sys.argv = ["streamlit", "run", "streamlit_app.py", *sys.argv[1:]]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
