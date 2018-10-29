###
###This is a Powershell Script for deploying the Azure shared resources required to support the Azure ML 
###demonstration cluster except for the container registry and containers. 
###
###It is intended to run from the root directory of the GitHub repository and will generate the credentials
###needed to run the 
###It is also intended to provide the parameters needed to build the Orchestration Driver
###config file and launch the rest of the deployment infrastructure.
###
###Note this script is intended to be run from the root directory of the AZMLSI project, and will not work properly otherwise
###

#Connect to the Azure gov't cluster, asuming the necessary dependencies have been installed
#https://docs.microsoft.com/en-us/azure/azure-government/documentation-government-get-started-connect-with-ps

Connect-AzureRmAccount -EnvironmentName AzureUSGovernment

####Build parameter section
#General Account Parameters
$location = "usgovarizona" #Set to Arizona in order to gain access to the most recent version of the Azure software
$resourceGroup = "SAIML" 

#Storage Account permissions
$storageAccountName = "saimldiag889" #Note this uses an existing BAH GOVT cluster account because there are no permissions to create one
$skuName = "Standard_LRS"

#DB Permissions
$locations = @(@{"locationName"="US Gov Arizona"; 
				"failoverPriority"=0}) 
$DBName = "azmlsidb"                              
$consistencyPolicy = @{"defaultConsistencyLevel"="BoundedStaleness";
                        "maxIntervalInSeconds"="10"; 
                        "maxStalenessPrefix"="200"}
$Capability="EnableCassandra"
$capabilities= @(@{"name"=$Capability})
$DBProperties = @{"databaseAccountOfferType"="Standard"; 
                           "locations"=$locations; 
                           "consistencyPolicy"=$consistencyPolicy;
                           "capabilities"=$capabilities}
                           
#####Resource Generation Section
#!!!NOTE THIS CURRENTLY DOESN"T WORK ON BAH GOVT!!! Storage Account (https://docs.microsoft.com/en-us/azure/storage/common/storage-powershell-guide-full)
#$storageAccount = New-AzureRMStorageAccount -ResourceGroupName $resourceGroup -Name $storageAccountName -Location $location	-SkuName $skuName

#CosmoDB Cassandra API Generation
$cosmoDB = New-AzureRmResource -ResourceType "Microsoft.DocumentDb/databaseAccounts" -ApiVersion "2015-04-08" -ResourceGroupName $resourceGroup -Location $location -Name $DBName -PropertyObject $DBProperties

####Resource key retrieval (for the OrchestrationDriver config)
$storageAccountKey = (Get-AzureRmStorageAccountKey -ResourceGroupName $resourceGroup -Name $storageAccountName).Value[0]
$storageAccountKey | ConvertTo-Json -depth 100 | Out-File "./OrchestrationDriver/storKeys.json"
                     
$dbKeysJson = Invoke-AzureRmResourceAction -Action listKeys -ResourceType "Microsoft.DocumentDb/databaseAccounts" -ApiVersion "2015-04-08" -ResourceGroupName $resourceGroup -Name $DBName
$dbKeysJson | ConvertTo-Json -depth 100 | Out-File "./OrchestrationDriver/dbKeys.json"

####Generate the container registry
$containerRegistryName ="myContainerRegistry"

#Container Registry Commands (https://github.com/MicrosoftDocs/azure-docs/blob/master/articles/container-registry/container-registry-get-started-powershell.md)
$registry = New-AzureRMContainerRegistry -ResourceGroupName $resourceGroup -Name $containerRegistryName -EnableAdminUser -Sku Basic -location $location

####Build the kubernetes cluster for cluster deployment and key distribution
az group create --name AksKubernetesResourceGroup --location westus2 --output jsonc
az aks create --resource-group AksKubernetesResourceGroup --name AksKubernetes --agent-count 3 --generate -ssh-keys --output jsonc

#Download dependencies and run the OrchestrationDriver to prep the DB
pip3 install -r ./OrchestrationDriver/requirements.txt 
python3 ./OrchestrationDriver/OrchestrationDriver.py -dk ./OrchestrationDriver/dbKeys.json -dn $DBName -sk ./OrchestrationDriver/storKeys.json -sn $storageAccountName

#mv ./VisualizationServiceConfig.json ./VisualizationService/VisualizationServiceConfig.json
#mv ./InputServiceConfig.json ./InputService/InputServiceConfig.json
#mv ./TrainingClassificationServiceConfig.json ./TrainingClassificationService/TrainingClassificationServiceConfig.json


####Container Build and Registration Section
docker build ./VisualizationService -t azmlsivizserv:latest
#docker build ./InputService -t azmlsiinserv:latest
#docker build ./TrainingClassificationService -t azmlsitrainclassserv:latest
$creds.Password | docker login $registry.LoginServer -u $creds.Username --password-stdin
