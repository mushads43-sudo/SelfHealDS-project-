locals {
  name = "${var.project}-${var.environment}"
  tags = merge({
    project     = var.project
    environment = var.environment
    managed_by  = "terraform"
    owner       = "J059"
  }, var.tags)
}

# ACR names must be globally unique, 5-50 lowercase alphanumerics.
resource "random_string" "suffix" {
  length  = 5
  upper   = false
  special = false
}

resource "azurerm_resource_group" "main" {
  name     = "rg-${local.name}"
  location = var.location
  tags     = local.tags
}

# ---------------------------------------------------------------------------
# Azure Container Registry - stores the versioned SelfHealDS images
# Basic SKU: ~USD 5/month, 10 GiB storage (plenty for ~700 MB images)
# ---------------------------------------------------------------------------
resource "azurerm_container_registry" "acr" {
  name                = "${var.project}acr${random_string.suffix.result}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "Basic"
  admin_enabled       = false # AKS pulls with its managed identity, CI pushes with OIDC
  tags                = local.tags
}

# ---------------------------------------------------------------------------
# Azure Kubernetes Service - runs SelfHealDS + Prometheus/Grafana
# Free control-plane tier; you pay only for the B-series nodes.
# ---------------------------------------------------------------------------
resource "azurerm_kubernetes_cluster" "aks" {
  name                      = "aks-${local.name}"
  location                  = azurerm_resource_group.main.location
  resource_group_name       = azurerm_resource_group.main.name
  dns_prefix                = local.name
  kubernetes_version        = var.kubernetes_version
  sku_tier                  = "Free"
  automatic_upgrade_channel = "patch"
  node_resource_group       = "rg-${local.name}-nodes"

  default_node_pool {
    name                        = "system"
    vm_size                     = var.node_vm_size
    auto_scaling_enabled        = true
    min_count                   = var.node_min_count
    max_count                   = var.node_max_count
    os_disk_size_gb             = 30
    os_sku                      = "Ubuntu"
    temporary_name_for_rotation = "systemtmp"

    upgrade_settings {
      max_surge = "10%"
    }
  }

  identity {
    type = "SystemAssigned"
  }

  network_profile {
    network_plugin      = "azure"
    network_plugin_mode = "overlay"
    load_balancer_sku   = "standard"
  }

  tags = local.tags
}

# Let the cluster's kubelet identity pull images from ACR (no imagePullSecrets needed).
resource "azurerm_role_assignment" "aks_acr_pull" {
  scope                            = azurerm_container_registry.acr.id
  role_definition_name             = "AcrPull"
  principal_id                     = azurerm_kubernetes_cluster.aks.kubelet_identity[0].object_id
  skip_service_principal_aad_check = true
}

# ---------------------------------------------------------------------------
# GitHub Actions permissions (optional, when the OIDC principal is known)
# ---------------------------------------------------------------------------
resource "azurerm_role_assignment" "github_acr_push" {
  count                = var.github_actions_principal_id == "" ? 0 : 1
  scope                = azurerm_container_registry.acr.id
  role_definition_name = "AcrPush"
  principal_id         = var.github_actions_principal_id
}

resource "azurerm_role_assignment" "github_aks_user" {
  count                = var.github_actions_principal_id == "" ? 0 : 1
  scope                = azurerm_kubernetes_cluster.aks.id
  role_definition_name = "Azure Kubernetes Service Cluster User Role"
  principal_id         = var.github_actions_principal_id
}
