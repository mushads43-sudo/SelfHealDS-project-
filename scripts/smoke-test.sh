#!/usr/bin/env bash
# Post-deployment smoke test.
# Usage: bash scripts/smoke-test.sh http://selfhealds-j059.centralindia.cloudapp.azure.com [http://localhost:8000]
set -euo pipefail
APP_URL="${1:?usage: smoke-test.sh <app-url> [metrics-url]}"
METRICS_URL="${2:-}"

echo -n "Health endpoint ............ "; curl -fsS --max-time 10 "$APP_URL/_stcore/health"; echo
echo -n "UI shell served ............ "; curl -fsS --max-time 10 "$APP_URL/" | grep -qi "<title>streamlit</title>\|streamlit" && echo ok
if [[ -n "$METRICS_URL" ]]; then
  echo -n "Metrics endpoint ........... "; curl -fsS --max-time 10 "$METRICS_URL/metrics" | grep -q '^selfhealds_build_info' && echo ok
  curl -fsS "$METRICS_URL/metrics" | grep '^selfhealds_build_info'
fi
echo "SMOKE TEST PASSED"
