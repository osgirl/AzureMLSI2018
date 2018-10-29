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

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from ssl import PROTOCOL_TLSv1_2, CERT_REQUIRED

import argparse

import os

import json

from azure.cosmos.cosmos_client import CosmosClient


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
   
def generateCosmoDB():
    '''
    Programmatically generates the CosmoDB Cassandra instance and retrieves the credentials required to generate a
    client and impose the required structure
    '''
    cosmos_client.session

def generateCosmoDBStructure(cfg, db_name, db_key, ca_file_uri):
    '''
    Prepares the CosmoDB Cassandra instance for the demonstration by imposing the required keyspace and table structure
    '''
    
    endpoint_uri = db_name + '.cassandra.cosmosdb.azure.us'
    
    cfg = cfg['cosmoDBParams']
    if cfg['localDBEndpoint']:
        #Running the cluster from a local instance without security options engaged
        #https://www.digitalocean.com/community/tutorials/how-to-install-cassandra-and-run-a-single-node-cluster-on-ubuntu-14-04
        cluster = Cluster()
    else:
        #Cassandra connection options for the Azure CosmoDB with Cassandra API from the quickstart documentation on the portal page
        #grabbed my CA.pem from Mozilla https://curl.haxx.se/docs/caextract.html
        ssl_opts = {
            'ca_certs': ca_file_uri,
            'ssl_version': PROTOCOL_TLSv1_2,
            'cert_reqs': CERT_REQUIRED  # Certificates are required and validated
        }
        auth_provider = PlainTextAuthProvider(username=db_name, password=db_key)
        cluster = Cluster([endpoint_uri], port = 10350, auth_provider=auth_provider, ssl_options=ssl_opts)
        
    #Checks to ensure that the demonstration keyspace exists and creates it if not 
    session = cluster.connect()
    print("\nCreating Keyspace")
    if cfg['localDBEndpoint']:
        #Modified local keyspace settings
        session.execute('CREATE KEYSPACE IF NOT EXISTS ' + cfg['cosmoDBKeyspaceName'] + ' WITH replication = {\'class\' : \'SimpleStrategy\', \'replication_factor\' : 1 }')
        session.shutdown()
    else:
        #Default keyspace settings from Azure CosmoDB
        session.execute('CREATE KEYSPACE IF NOT EXISTS ' + cfg['cosmoDBKeyspaceName'] + ' WITH replication = {\'class\': \'NetworkTopologyStrategy\', \'datacenter\' : 1 }'); #Formatting is stupid on this string due to the additional curley braces
    
    session = cluster.connect(cfg['cosmoDBKeyspaceName']);   

    personaTableName = cfg['personaTableName']
    personaEdgeTableName = cfg['personaEdgeTableName']
    rawImageTableName = cfg['rawImageTableName']
    refinedImageTableName = cfg['refinedImageTableName']

    #Create table
    print("\nCreating Table")
    session.execute('DROP TABLE IF EXISTS ' + personaTableName)
    session.execute('DROP TABLE IF EXISTS ' + personaEdgeTableName)
    session.execute('DROP TABLE IF EXISTS ' + rawImageTableName)
    session.execute('DROP TABLE IF EXISTS ' + refinedImageTableName)

    #Persona table provides metadata on each persona in the demonstration database such as name and date of birth
    #This table is set up primarily to be scanned and used to pivot to the persona edge table to discover images 
    session.execute('CREATE TABLE IF NOT EXISTS ' + personaTableName + ' (persona_name text, PRIMARY KEY(persona_name))');
    
    #Persona edge table contains the associations to pivot from a selected persona to its associated images
    #These associations can exist either due to explicit labeling or predicted labels
    session.execute('CREATE TABLE IF NOT EXISTS ' + personaEdgeTableName + ' (persona_name text, assoc_image_id text, label_assoc_flag boolean, pred_assoc_flag boolean, PRIMARY KEY(persona_name, assoc_image_id))');
    
    #Refined table stores extracted face blobs and associative edges to the raw image from which it was derived
    session.execute('CREATE TABLE IF NOT EXISTS ' + refinedImageTableName + ' (image_id text, raw_image_edge_id text, image_bytes blob, PRIMARY KEY(image_id))');
    
    #Raw table stores pre-extraction images that contain at least one face
    session.execute('CREATE TABLE IF NOT EXISTS ' + rawImageTableName + ' (image_id text, refined_image_edge_id text, file_uri text, image_bytes blob, PRIMARY KEY(image_id))');


def generateServiceConfigurationFiles(cfg, db_key, db_name, stor_key, stor_name):
    '''
        Uses the master configuration file for the orchestration driver with additional information generated from the
        resource generation process to produce the VisualizationService, InputService and RetrainingService config files to be bundled
        with their docker container
    '''
    #Add argument parameters to the json dictionaries for the services
    cfg['cosmoDBParams']['cosmoDBName'] = db_name
    cfg['cosmoDBParams']['cosmoDBKey'] = db_key
    cfg['cosmoDBParams']['storageName'] = stor_name
    cfg['cosmoDBParams']['storageKey'] = stor_key
    
    input_service_config_dict = {
        'AzureFileStorageParams':cfg['AzureFileStorageParams'], 
        'cosmoDBParams':cfg['cosmoDBParams']
    }
    with open('./InputServiceConfig.json', 'w') as write_file:
        json.dump(input_service_config_dict, write_file)

    input_service_config_dict = {
        'cosmoDBParams':cfg['cosmoDBParams']
    }
    with open('./VisualizationServiceConfig.json', 'w') as write_file:
        json.dump(input_service_config_dict, write_file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dk', nargs=1)
    parser.add_argument('-dn', nargs=1)
    parser.add_argument('-sk', nargs=1)
    parser.add_argument('-sn')
    args = parser.parse_args()
    
    cfg = json.load(open('./OrchestrationDriver/OrchestrationDriverConfig.json', 'r'))
    db_key = json.load(open(args.dk[0], 'r'))['primaryMasterKey']
    db_name = args.dn[0]
    stor_key = json.load(open(args.sk[0], 'r'))
    stor_name = args.sn[0]
    ca_file_uri = "./OrchestrationDriver/cacert.pem"
    
    print(db_key)
    print(stor_key)
    #generateCosmoDBStructure(cfg, db_name, db_key, ca_file_uri)
    generateServiceConfigurationFiles(cfg, db_key, db_name, stor_key, stor_name)
    

if __name__ == '__main__':
    main()
    
    #generateCosmoDBStructure(cfg)
    #generateServiceConfigurationFiles(cfg)
    '''
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