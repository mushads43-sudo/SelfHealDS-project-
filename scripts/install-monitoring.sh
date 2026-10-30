#!/usr/bin/env bash
# Install/upgrade kube-prometheus-stack and the SelfHealDS monitoring objects.
# Requires: kubectl context pointing at the AKS cluster, helm 3.
# Usage: GRAFANA_ADMIN_PASSWORD='choose-one' bash scripts/install-monitoring.sh
set -euo pipefail

: "${GRAFANA_ADMIN_PASSWORD:?set GRAFANA_ADMIN_PASSWORD}"
CHART_VERSION="${CHART_VERSION:-65.1.1}"

helm repo add prometheus-community https://prometheus-community.github.io/helm-charts >/dev/null 2>&1 || true
helm repo update >/dev/null

helm upgrade --install monitoring prometheus-community/kube-prometheus-stack \
  --version "$CHART_VERSION" \
  --namespace monitoring --create-namespace \
  -f k8s/monitoring/kube-prometheus-stack-values.yaml \
  --set grafana.adminPassword="$GRAFANA_ADMIN_PASSWORD" \
  --wait --timeout 10m

# The selfhealds namespace must exist before namespaced monitoring objects.
kubectl get namespace selfhealds >/dev/null 2>&1 || kubectl create namespace selfhealds
kubectl apply -k k8s/monitoring

echo
kubectl -n monitoring get pods
echo
echo "Grafana:    kubectl -n monitoring port-forward svc/monitoring-grafana 3000:80   -> http://localhost:3000 (admin)"
echo "Prometheus: kubectl -n monitoring port-forward svc/monitoring-kube-prometheus-prometheus 9090:9090"
