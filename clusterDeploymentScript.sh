az cloud set --name AzureUSGovernment
az login
az cloud list --output table
az group create --name AcsKubernetesResourceGroup --location usgovvirginia --tags orchestrator=kubernetes
az acs create --orchestrator-type kubernetes --name AcsKubernetes --resource-group AcsKubernetesResourceGroup --generate-ssh-keys --output jsonc