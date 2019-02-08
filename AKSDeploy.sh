#sudo apt install jq
#sudo snap install yq

RESOURCE_GROUP="CommercialCyberAzure"
CONTAINER_ACCOUNT="AZMLSIACR"
URL_CONTAINER_ACCOUNT="azmlsiacr" 
## needs to be lower case

#Export in order to make available to scripts
export COSMOS_DB_ACCOUNT="azmlsidb"
export BLOB_STORAGE_ACCOUNT="commercialcyberazurediag"
export BLOB_CONTAINER_NAME="test-data-con"


COG_SERV_NAME="azmlsi_face_matcher"
#LOCATION="usgovarizona"
LOCATION="eastus"

#
CAPABILITIES="EnableCassandra"
CONSISTENCY="BoundedStaleness"
MAXINTERVALINSECONDS="10"
MAXSTALENESSPREFIX="200"
FAILOVERPRIORITY="false"
KIND="GlobalDocumentDB"

AKSNAME="k8s-server"

#Login local cloud shell to Azure gov't
#az cloud set --name AzureUSGovernment
#az cloud set --name AzureCloud
#az login

## Create group
#az group create --name $RESOURCE_GROUP --location $LOCATION

#Generate new Azure CosmosDB with Cassandra API
#az cosmosdb create --name $COSMOS_DB_ACCOUNT --resource-group $RESOURCE_GROUP --capabilities $CAPABILITIES --default-consistency-level $CONSISTENCY --max-interval $MAXINTERVALINSECONDS --max-staleness-prefix $MAXSTALENESSPREFIX --enable-automatic-failover $FAILOVERPRIORITY --kind $KIND
#az cosmosdb delete --name $COSMOS_DB_ACCOUNT --resource-group $RESOURCE_GROUP
export COSMOS_DB_KEY=`az cosmosdb list-keys -n $COSMOS_DB_ACCOUNT -g $RESOURCE_GROUP | jq -r '."primaryMasterKey"'`

#Generate new Azure Blob Storage 
#az storage blob create --name $BLOB_CONTAINER_NAME
#az storage blob delete --name $BLOB_CONTAINER_NAME
export BLOB_STORAGE_KEY=`az storage account keys list -g $RESOURCE_GROUP -n $BLOB_STORAGE_ACCOUNT | jq -r '.[]| select(.keyName == "key1")|.value'`
#az storage container create --name $BLOB_CONTAINER_NAME --account-name $BLOB_STORAGE_ACCOUNT --account-key $BLOB_STORAGE_KEY
#Edit the cluster configuration file
#yq w -i -d1 cluster-deployment.yml 'data.blob-storage-con' $BLOB_CONTAINER_NAME

#Generate new Azure Cognitive Services API for facial detection
#az cognitiveservices account create --name $COG_SERV_NAME --resource-group $RESOURCE_GROUP --kind Face --sku S0 -l $LOCATION
#az cognitiveservices account delete --name $COG_SERV_NAME --resource-group $RESOURCE_GROUP
COG_SERV_KEY=`az cognitiveservices account keys list --name $COG_SERV_NAME --resource-group $RESOURCE_GROUP -o json | jq -r '.key1'`

#Generate an Azure Container Registry and extract access credentials
#az acr create --name $CONTAINER_ACCOUNT --resource-group $RESOURCE_GROUP --sku Standard --location $LOCATION
#az acr update --name $CONTAINER_ACCOUNT --admin-enabled true
#az acr delete --name $CONTAINER_ACCOUNT --resource-group $RESOURCE_GROUP
ACR_LOGIN_USERNAME=`az acr credential show --name $CONTAINER_ACCOUNT --output json | jq -r '.username'`
ACR_LOGIN_KEY=`az acr credential show --name $CONTAINER_ACCOUNT --output json | jq -r '.passwords|.[0]|.value'`

#Get eventhub credential string
export EH_KEY=`az eventhubs namespace authorization-rule keys list --resource-group CommercialCyberAzure --namespace azmlsieh --name RootManageSharedAccessKey | jq -r '.primaryKey'`
export EH_ACCOUNT=`az eventhubs namespace authorization-rule keys list --resource-group CommercialCyberAzure --namespace azmlsieh --name RootManageSharedAccessKey | jq -r '.keyName'`
export EH_NAMESPACE=azmlsieh
export EH_NAME=testhub
export EH_URL="://azmlsieh.servicebus.windows.net/"

#Build and deploy containers to Azure Container Registry
#ACR_URI="$CONTAINER_ACCOUNT.azurecr.us" #For gov't
ACR_URI="$URL_CONTAINER_ACCOUNT.azurecr.io" #For Commercial
#sudo docker build -t "$ACR_URI/input-service" ./InputService/
#sudo docker build -t "$ACR_URI/viz-service" ./VisualizationService/
#sudo docker login $ACR_URI -u $ACR_LOGIN_USERNAME -p $ACR_LOGIN_KEY
#sudo docker push "$ACR_URI/input-service"
#sudo docker push "$ACR_URI/viz-service"

#Setup environment variables, install dependencies and run Orchestration Driver
export CA_FILE_URI="./OrchestrationDriver/cacert.pem"
export CLUSTER_CONFIG_URI="cluster-deployment.yml"

#sudo pip3 install -r ./OrchestrationDriver/requirements.txt
#python3 ./OrchestrationDriver/OrchestrationDriver.py

#Register Azure secrets to prep for deployment
CR_SECRET_NAME=image-pull-secrets
YOUR_MAIL=wharton_caleb@bah.com
BLOB_SECRET_NAME=bs-secrets
DB_SECRET_NAME=db-secrets
CS_SECRET_NAME=cs-secrets
EH_SECRET_NAME=eh-secrets
#Login to Azure Kubernetes cluster
#az aks get-credentials --resource-group $RESOURCE_GROUP --name $AKSNAME

## K8s cluster deployment configuration 
export K8_CLUSTER_CONFIG="./cluster-deployment.yml"


#Secret for Kubernetes Azure Container Registry
kubectl create secret docker-registry $CR_SECRET_NAME --docker-server $ACR_URI --docker-email $YOUR_MAIL --docker-username=$ACR_LOGIN_USERNAME --docker-password $ACR_LOGIN_KEY
kubectl create secret generic $BLOB_SECRET_NAME --from-literal=blob-storage-account=$BLOB_STORAGE_ACCOUNT --from-literal=blob-storage-key=$BLOB_STORAGE_KEY
kubectl create secret generic $DB_SECRET_NAME --from-literal=db-account=$COSMOS_DB_ACCOUNT --from-literal=db-key=$COSMOS_DB_KEY
kubectl create secret generic $CS_SECRET_NAME --from-literal=cs-account=$COG_SERV_NAME --from-literal=cs-key=$COG_SERV_KEY
kubectl create secret generic $EH_SECRET_NAME --from-literal=eh-url=$EH_NAMESPACE$EH_URL$EH_NAME --from-literal=eh-account=$EH_ACCOUNT --from-literal=eh-key=$EH_KEY
kubectl create -f $K8_CLUSTER_CONFIG


#az aks create --resource-group $RESOURCE_GROUP --name $AKS_NAME --node-count 1 --enable-addons monitoring --generate-ssh-keys
