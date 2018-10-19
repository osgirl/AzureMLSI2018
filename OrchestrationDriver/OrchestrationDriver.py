'''
Deployment script for:

* Starting up the cluster shared resources  
* Loading the demonstration Test Data into Azure File Storage
* Building the Demonstration System Containers and deploying to Azure Storage
* Starting the Demonstration System Containers on Azure to initialize the cluster

Resources
https://github.com/Azure-Samples/container-instances-python-manage    

'''
from utilities import AzureContext
from azure.storage.file import FileService
from azure.storage.file import ContentSettings
from azure.common.credentials import ServicePrincipalCredentials

from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.containerinstance import ContainerInstanceManagementClient
from azure.mgmt.containerinstance.models import (ContainerGroup, Container, ContainerPort, Port, IpAddress, 
                                                 ResourceRequirements, ResourceRequests, ContainerGroupNetworkProtocol, OperatingSystemTypes)
#from azure.mgmt.containerregistry import ContainerRegistryManagementClient
from azure.mgmt.containerregistry.v2018_09_01.container_registry_management_client import ContainerRegistryManagementClient
from azure.mgmt.containerregistry.v2018_09_01.models import Registry
from azure.mgmt.containerregistry.v2018_09_01.models import ImportImageParameters, ImportSource

import argparse
import os

import OrchestrationDriverConfig as cfg


def cleanupAccountDir(fileService, shareName, dirName):
    '''
    
    '''
    for file in fileService.list_directories_and_files(shareName, dirName):
        print(file.name)
        fileService.delete_file(shareName, dirName, file.name)
    fileService.delete_directory(shareName, dirName)

def buildDockerContainer():
    '''     
        Runs the Docker build command against the prewritten docker files within the repository to 
        generate the containers for each cluster component and prepare them for transport to
        the cluster
         
        https://medium.com/@bmihelac/examples-for-building-docker-images-with-private-python-packages-6314440e257c
        https://github.com/docker/docker-py
    '''
    
    def buildFrontendContainer():
        return None
    
    def buildInitContainer():
        return None
    
    
    return None

def deployDockerContainer():
    '''
        Takes the built docker images from the previous function and places them in the Azure Azure Container 
        Registry in preperation for instantiation as part of the cluster overall
    
    '''
    
    return None

def create_container_group(client, demoResourceGroupName, name, location, image, memory, cpu):

   # setup default values
   port = 80
   container_resource_requirements = None
   command = None
   environment_variables = None

   # set memory and cpu
   container_resource_requests = ResourceRequests(memory_in_gb = memory, cpu = cpu)
   container_resource_requirements = ResourceRequirements(requests = container_resource_requests)
   
   container = Container(name = name,
                         image = image,
                         resources = container_resource_requirements,
                         command = command,
                         ports = [ContainerPort(port=port)],
                         environment_variables = environment_variables)

   # defaults for container group
   cgroup_os_type = OperatingSystemTypes.linux
   cgroup_ip_address = IpAddress(type='public', ports = [Port(protocol=ContainerGroupNetworkProtocol.tcp, port = port)])
   image_registry_credentials = None
   cgroup = ContainerGroup(location = location, containers = [container], os_type = cgroup_os_type, ip_address = cgroup_ip_address, image_registry_credentials = image_registry_credentials)
   client.container_groups.create_or_update(demoResourceGroupName, name, cgroup)

def delete_resources(demoResourceGroupName, demoContainerGroupName): 
   client.container_groups.delete(demoResourceGroupName, demoContainerGroupName)
   resource_client.resource_groups.delete(demoResourceGroupName)

if __name__ == '__main__':

    fileService = FileService(account_name=cfg.fileStorageAccountName, account_key=cfg.fileStorageSecret)
    fileService.create_share(cfg.fileStorageShareName)
    fileService.create_directory(cfg.fileStorageShareName, cfg.fileStorageDir)
    
    #This will arbitrarily copy the contents of a system folder to the Azure file storage
    for dirPath, dirNames, fileNames in os.walk("testDataFolder"):
        for fileName in fileNames:
            fileHandle = open(dirPath + "/" + fileName) #TODO make more robust with a non-OS specific seperator
            fileService.create_file_from_path(cfg.fileStorageShareName, cfg.fileStorageDir, fileName, dirPath + "/" + fileName)
            fileHandle.close()
    
    cleanupAccountDir(fileService, cfg.fileStorageShareName, cfg.fileStorageDir)
    
    #Uses a bit of stub code provided by Azure with credentials collected using the following
    #https://docs.microsoft.com/en-us/azure/azure-resource-manager/resource-group-create-service-principal-portal
    azure_context = AzureContext(
        subscription_id = cfg.azureSubscriptionId,
        client_id = cfg.servicePrincipalClientId,
        client_secret = cfg.servicePrincipalClientKey,
        tenant = cfg.servicePrincipalTenantId
    )

    #Generates the required clients using the azure context from above
    resource_client = ResourceManagementClient(azure_context.credentials, azure_context.subscription_id)
    client = ContainerInstanceManagementClient(azure_context.credentials, azure_context.subscription_id)
    registryClient = ContainerRegistryManagementClient(azure_context.credentials, azure_context.subscription_id)
    
    location = 'eastus'

    #Generate resource group and adds a container group to it in preperation for the cluster launch
    resource_client.resource_groups.create_or_update(cfg.demoResourceGroupName, {'location': location})
    create_container_group(client = client, demoResourceGroupName = cfg.demoResourceGroupName, 
        name = cfg.demoContainerGroupName, 
        location = location, 
        image = "microsoft/aci-helloworld", 
        memory = 1, 
        cpu = 1) 
        
    cgroup = client.container_groups.get(cfg.demoResourceGroupName, cfg.demoContainerGroupName)
    print("{0}, {1}".format(cgroup.name, cgroup.location))
    
    #Cleanup resource and container groups
    client.container_groups.delete(cfg.demoResourceGroupName, cfg.demoContainerGroupName)
    resource_client.resource_groups.delete(cfg.demoResourceGroupName)
    
    '''
    print(registryClient.registries.check_name_availability(cfg.demoRegistryGroupName))
    for registry in registryClient.registries.list():
        print(registry)
        
        #This is somewhat sticky as the actual location of the docker file depends on the storage driver used by
        #the local instance of docker
        ImportImageParameters(ImportSource())
        registryClient.registries.import_image(cfg.demoResourceGroupName, cfg.demoRegistryGroupName, parameters)
        
    #Turns out it is EXTREMELY annoying to create a new registry programmatically due to the need to completely
    #fill out the attributes https://portal.azure.com/#create/Microsoft.ContainerRegistry
    
    #registryClient.registries.create(demoResourceGroupName, registry_name)
    '''