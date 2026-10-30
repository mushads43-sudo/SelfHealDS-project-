# SelfHealDS - MLOps quick reference

The full step-by-step guide is `SelfHealDS_MLOps_Implementation_Guide.docx`.
The research code (`app.py`, `backend/`) is unchanged; the MLOps layer wraps it.

| Stage | Where | Run |
|---|---|---|
| Version control | `.gitignore`, branches `main` / `develop` / `feature/*`, tags `vX.Y.Z` | `git flow` style, see guide section 1 |
| CI | `.github/workflows/ci.yml` | every PR / push to develop |
| Quality gate | `mlops/validate_model.py`, `mlops/thresholds.toml` | `make gate` |
| Docker | `Dockerfile`, `.dockerignore`, `docker-compose.yml` | `make docker-build docker-run`, `make compose-up` |
| Kubernetes | `k8s/base/*` (kustomize) | `kubectl apply -k k8s/base` |
| Terraform | `infra/terraform/*` | `make tf-init tf-plan tf-apply` |
| Monitoring | `monitoring/` (app metrics), `k8s/monitoring/`, `observability/` | `make monitoring` |
| CD to Azure | `.github/workflows/cd.yml` | push to `main` or tag `v*.*.*` |

Entry points:

- `streamlit run app.py` - original research UI, no monitoring.
- `python -m monitoring.launch` - production: same UI + Prometheus `/metrics` on port 8000.

Cost control: `make aks-stop` after a demo, `make aks-start` before; `make tf-destroy` when finished.
