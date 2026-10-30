variable "subscription_id" {
  description = "Azure subscription ID (az account show --query id -o tsv)"
  type        = string
}

variable "project" {
  description = "Short project name used in resource names"
  type        = string
  default     = "selfhealds"
}

variable "environment" {
  description = "Deployment environment tag"
  type        = string
  default     = "prod"
}

variable "location" {
  description = "Azure region. centralindia is closest to Mumbai; if your subscription policy blocks it, try southindia or eastus."
  type        = string
  default     = "centralindia"
}

variable "kubernetes_version" {
  description = "AKS Kubernetes version; null = current AKS default"
  type        = string
  default     = null
}

variable "node_vm_size" {
  description = "Node size. Standard_B2s (2 vCPU / 4 GiB) is the cheapest size AKS accepts for a system pool; use Standard_B2ms (8 GiB) if pods stay Pending."
  type        = string
  default     = "Standard_B2s"
}

variable "node_min_count" {
  description = "Cluster-autoscaler minimum nodes"
  type        = number
  default     = 1
}

variable "node_max_count" {
  description = "Cluster-autoscaler maximum nodes (2 x B2s = 4 vCPU fits the Azure for Students regional quota)"
  type        = number
  default     = 2

  validation {
    condition     = var.node_max_count >= var.node_min_count
    error_message = "node_max_count must be >= node_min_count."
  }
}

variable "github_actions_principal_id" {
  description = "Object ID of the GitHub Actions service principal (scripts/setup-github-oidc.sh prints it). Empty = skip CI role assignments."
  type        = string
  default     = ""
}

variable "tags" {
  description = "Extra tags applied to every resource"
  type        = map(string)
  default     = {}
}
