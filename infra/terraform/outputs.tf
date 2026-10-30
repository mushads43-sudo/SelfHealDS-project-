output "resource_group_name" {
  value = azurerm_resource_group.main.name
}

output "acr_name" {
  value = azurerm_container_registry.acr.name
}

output "acr_login_server" {
  description = "Use as the image prefix: <login_server>/selfhealds:<tag>"
  value       = azurerm_container_registry.acr.login_server
}

output "aks_cluster_name" {
  value = azurerm_kubernetes_cluster.aks.name
}

output "get_credentials_command" {
  value = "az aks get-credentials --resource-group ${azurerm_resource_group.main.name} --name ${azurerm_kubernetes_cluster.aks.name} --overwrite-existing"
}

output "app_url_hint" {
  description = "Public URL once the LoadBalancer Service exists (label set in k8s/base/service.yaml)"
  value       = "http://<azure-dns-label-name>.${azurerm_resource_group.main.location}.cloudapp.azure.com"
}
