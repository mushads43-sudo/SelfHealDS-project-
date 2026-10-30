#!/usr/bin/env bash
# One-time: create an Azure AD app + service principal that GitHub Actions
# uses through OpenID Connect (no client secret is ever stored in GitHub).
# Usage: bash scripts/setup-github-oidc.sh <github-owner>/<repo>
#   e.g. bash scripts/setup-github-oidc.sh mushads43-sudo/selfhealds
set -euo pipefail

REPO="${1:?usage: setup-github-oidc.sh <owner>/<repo>}"
APP_NAME="github-selfhealds-cicd"
SUBSCRIPTION_ID="$(az account show --query id -o tsv)"
TENANT_ID="$(az account show --query tenantId -o tsv)"

APP_ID="$(az ad app create --display-name "$APP_NAME" --query appId -o tsv)"
az ad sp create --id "$APP_ID" --output none 2>/dev/null || true
SP_OBJECT_ID="$(az ad sp show --id "$APP_ID" --query id -o tsv)"

# Federated credentials: one per GitHub OIDC "subject" the workflows use.
add_cred () {
  az ad app federated-credential create --id "$APP_ID" --parameters "{
    \"name\": \"$1\", \"issuer\": \"https://token.actions.githubusercontent.com\",
    \"subject\": \"$2\", \"audiences\": [\"api://AzureADTokenExchange\"] }" --output none
}
add_cred "env-production"     "repo:${REPO}:environment:production"      # cd.yml, monitoring.yml
add_cred "env-infrastructure" "repo:${REPO}:environment:infrastructure"  # terraform apply
add_cred "pull-request"       "repo:${REPO}:pull_request"                # terraform plan on PRs

# Terraform creates resource groups AND role assignments, so it needs
# Contributor + User Access Administrator at subscription scope.
az role assignment create --assignee-object-id "$SP_OBJECT_ID" --assignee-principal-type ServicePrincipal \
  --role Contributor --scope "/subscriptions/$SUBSCRIPTION_ID" --output none
az role assignment create --assignee-object-id "$SP_OBJECT_ID" --assignee-principal-type ServicePrincipal \
  --role "User Access Administrator" --scope "/subscriptions/$SUBSCRIPTION_ID" --output none

echo
echo "Add these GitHub SECRETS (Settings > Secrets and variables > Actions):"
echo "  AZURE_CLIENT_ID=$APP_ID"
echo "  AZURE_TENANT_ID=$TENANT_ID"
echo "  AZURE_SUBSCRIPTION_ID=$SUBSCRIPTION_ID"
echo "and this repository VARIABLE:"
echo "  GITHUB_SP_OBJECT_ID=$SP_OBJECT_ID"
echo
echo "Or with the GitHub CLI:"
echo "  gh secret set AZURE_CLIENT_ID -b $APP_ID -R $REPO"
echo "  gh secret set AZURE_TENANT_ID -b $TENANT_ID -R $REPO"
echo "  gh secret set AZURE_SUBSCRIPTION_ID -b $SUBSCRIPTION_ID -R $REPO"
echo "  gh variable set GITHUB_SP_OBJECT_ID -b $SP_OBJECT_ID -R $REPO"
