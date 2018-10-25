###
###This is a Powershell Script for deploying the Azure shared resources required to support the Azure ML 
###demonstration cluster. It is also intended to provide the parameters needed to build the Orchestration Driver
###config file and launch the rest of the deployment infrastructure
###

####Build parameter section
#General Account Parameters
$location = "usgovarizona" #Set to Arizona in order to gain access to the most recent version of the Azure software
$resourceGroup = "SAIML" 

#Storage Account permissions
$storageAccountName = "azmlsifilestorage"

#DB Permissions
$locations = @(@{"locationName"="US Gov Virginia";                        "failoverPriority"=0}) 
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
                           
$containerRegistryName ="myContainerRegistry"

#####Resource Generation Section
#Storage Account
$storageAccount = New-AzureRMStorageAccount -ResourceGroupName $resourceGroup `
	-Name $storageAccountName `
	-Location $location `
	-SkuName $skuName

#Container Registry Commands
$registry = New-AzureRMContainerRegistry -ResourceGroupName $resourceGroup -Name $containerRegistryName -EnableAdminUser -Sku Basic -location $location
$creds = Get-AzureRmContainerRegistryCredential -Registry $registry #Login
$creds.Password | docker login $registry.LoginServer -u $creds.Username --password-stdin


#CosmoDB Cassandra API Generation
$cosmoDB = New-AzureRmResource -ResourceType "Microsoft.DocumentDb/databaseAccounts" -ApiVersion "2015-04-08" -ResourceGroupName $resourceGroup -Location $location -Name $DBName -PropertyObject $DBProperties
                     
                     
####Resource key retrieval (for the OrchestrationDriver config)

$storageAccountKey = (Get-AzureRmStorageAccountKey -ResourceGroupName $resourceGroup -Name $storageAccountName).Value[0]
                     
$dbKeysJson = Invoke-AzureRmResourceAction -Action listKeys -ResourceType "Microsoft.DocumentDb/databaseAccounts" -ApiVersion "2015-04-08" -ResourceGroupName $resourceGroup -Name $DBName
$dbKeysJson | ConvertTo-Json -depth 100 | Out-File ".\file.json"