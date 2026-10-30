terraform {
  required_version = ">= 1.6.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 5.11"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }

  # Remote state in Azure Blob Storage (created once by scripts/bootstrap-tfstate.sh).
  # Values are supplied at init time:  terraform init -backend-config=backend.hcl
  backend "azurerm" {}
}

provider "azurerm" {
  features {
    resource_group {
      # Allow `terraform destroy` even if AKS created extra resources in the RG.
      prevent_deletion_if_contains_resources = false
    }
  }
  subscription_id = var.subscription_id
}
