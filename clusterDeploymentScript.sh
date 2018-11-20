RESOURCE_GROUP="SAIML"
COSMO_DB_NAME="azmlsidb"
STORAGE_ACCOUNT_NAME="saimldiag889"
BLOB_CONTAINER_NAME="test-data-con"
CONTAINER_ACCOUNT="mycontainerregistry"
COG_SERV_NAME="azmlsi_face_matcher"


#Login local cloud shell to Azure gov't
az cloud set --name AzureUSGovernment
az login

#Generate new Azure CosmosDB with Cassandra API
az cosmosdb create --name $COSMO_DB_NAME --resource-group $RESOURCE_GROUP --capabilities EnableCassandra
az cosmosdb list-keys --name $COSMO_DB_NAME --resource-group $RESOURCE_GROUP -o YAML > db_keys.yaml

#Generate new Azure Cognitive Services API for facial detection
az cognitiveservices account create --name $COG_SERV_NAME --resource-group $RESOURCE_GROUP --kind Face --sku S0 -l usgovvirginia
az cognitiveservices account keys list --name $COG_SERV_NAME --resource-group $RESOURCE_GROUP -o yaml > cog_serv_key.yaml

#Generate new Azure Blob Storage 
az storage blob create --name $BLOB_CONTAINER_NAME
az storage account keys list --account-name $STORAGE_ACCOUNT_NAME --output YAML > bs_keys.yaml

#Generate an Azure Container Registry and extract access credentials
az acr create --name $CONTAINER_ACCOUNT --resource-group SAIML --sku Standard --location usgovarizona
az acr update --name $CONTAINER_ACCOUNT --admin-enabled true
az acr credential show --name $CONTAINER_ACCOUNT --output yaml > acr_keys.yaml

#Build and deploy containers to Azure Container Registry
ACR_URI="{$CONTAINER_ACCOUNT}.azurecr.us"
sudo docker login $ACR_URI
sudo docker build -t "{$ACR_URI}/input-service" ./InputService/
sudo docker build -t "{$ACR_URI}/viz-service" ./VisualizationService/
sudo docker push "{$ACR_URI}/input-service"
sudo docker push "{$ACR_URI}/viz-service"

#az cloud list --output table
#az group create --name AcsKubernetesResourceGroup --location usgovvirginia --tags orchestrator=kubernetes
#az acs create --orchestrator-type kubernetes --name AcsKubernetes --resource-group AcsKubernetesResourceGroup --generate-ssh-keys --output jsonc