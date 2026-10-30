#!/usr/bin/env bash
# One-time: create the Azure Storage account that holds Terraform remote state.
# Usage: bash scripts/bootstrap-tfstate.sh [location]
set -euo pipefail

LOCATION="${1:-centralindia}"
RG="rg-selfhealds-tfstate"
SA="selfhealdstf$(openssl rand -hex 3)"   # globally unique, lowercase, <=24 chars
CONTAINER="tfstate"

az group create --name "$RG" --location "$LOCATION" --output none
az storage account create --name "$SA" --resource-group "$RG" --location "$LOCATION" \
  --sku Standard_LRS --kind StorageV2 --min-tls-version TLS1_2 \
  --allow-blob-public-access false --output none
az storage container create --name "$CONTAINER" --account-name "$SA" --auth-mode login --output none 2>/dev/null \
  || az storage container create --name "$CONTAINER" --account-name "$SA" --output none

cat > infra/terraform/backend.hcl <<EOF
resource_group_name  = "$RG"
storage_account_name = "$SA"
container_name       = "$CONTAINER"
key                  = "selfhealds.prod.tfstate"
EOF

echo "Terraform state storage ready."
echo "  Written: infra/terraform/backend.hcl"
echo "  Set GitHub repository variables:"
echo "    TFSTATE_RESOURCE_GROUP=$RG"
echo "    TFSTATE_STORAGE_ACCOUNT=$SA"
