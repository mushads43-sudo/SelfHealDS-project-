# SelfHealDS - common developer / MLOps commands.  Run `make help`.
IMAGE      ?= selfhealds
TAG        ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo dev)
ACR        ?= selfhealdsacr
TF_DIR     := infra/terraform

.PHONY: help install lint test gate run run-prod docker-build docker-run compose-up compose-down \
        k8s-render k8s-validate tf-init tf-plan tf-apply tf-destroy deploy monitoring aks-stop aks-start

help:            ## Show targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-14s %s\n",$$1,$$2}'

install:         ## Install app + dev dependencies into the active venv
	pip install -r requirements-dev.txt

lint:            ## Ruff (CI policy)
	ruff check .

test:            ## Unit tests with coverage
	pytest -q --cov=backend --cov=monitoring --cov-report=term

gate:            ## Model quality gate (controlled-corruption benchmark)
	python mlops/validate_model.py --report mlops/reports/quality_gate.json

run:             ## Original research UI (unchanged)
	streamlit run app.py

run-prod:        ## Production entry point with /metrics on :8000
	python -m monitoring.launch

docker-build:    ## Build image tagged with the git sha
	docker build -t $(IMAGE):$(TAG) --build-arg APP_VERSION=$(TAG) --build-arg GIT_SHA=$(TAG) .

docker-run:      ## Run image like Kubernetes does (read-only, non-root)
	docker run --rm -p 8501:8501 -p 8000:8000 --read-only --tmpfs /tmp \
	  --tmpfs /home/app:uid=10001,gid=10001 $(IMAGE):$(TAG)

compose-up:      ## App + Prometheus + Grafana locally
	docker compose up --build -d

compose-down:
	docker compose down

k8s-render:      ## Show the rendered Kubernetes manifests
	kubectl kustomize k8s/base

k8s-validate:    ## Schema-validate manifests (needs kubeconform)
	kubectl kustomize k8s/base | kubeconform -strict -summary

tf-init:
	cd $(TF_DIR) && terraform init -backend-config=backend.hcl
tf-plan:
	cd $(TF_DIR) && terraform plan -out=tfplan
tf-apply:
	cd $(TF_DIR) && terraform apply tfplan
tf-destroy:      ## Delete ALL Azure resources (stops billing)
	cd $(TF_DIR) && terraform destroy

deploy:          ## Manual deploy of TAG to AKS
	cd k8s/base && kustomize edit set image selfhealds=$(ACR).azurecr.io/$(IMAGE):$(TAG)
	kubectl apply -k k8s/base && kubectl -n selfhealds rollout status deploy/selfhealds

monitoring:      ## Install kube-prometheus-stack + dashboards/alerts
	bash scripts/install-monitoring.sh

aks-stop:        ## Pause the cluster (no node charges) - run after a demo
	az aks stop -g rg-selfhealds-prod -n aks-selfhealds-prod
aks-start:
	az aks start -g rg-selfhealds-prod -n aks-selfhealds-prod
