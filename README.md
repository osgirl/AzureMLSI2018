# Azure ML Model Deployment and Sustainment SI 2018
## Introduction
GitHub project for an information processing system which provides an example of how 
to design a modern ML processing system on public cloud architecture. Specifically 
this system is designed to demonstrate several key DevOps concepts related to agile 
systems and particularly ML systems

## Deployment Instructions

### Azure Setup For Ubuntu 16.04

1. Navigate to the project root directory in the terminal/CLI
1. Acquire an Azure subscription on the commercial or gov't cluster with the permissions 
   needed to generate Storage and CosmosDB accounts
1. Setup powershell in your local host environment
   (https://docs.microsoft.com/en-us/powershell/scripting/setup/installing-powershell-core-on-linux?view=powershell-6#snap-package)
1. Configure the local powershell for Azure
   (https://docs.microsoft.com/en-us/powershell/azure/install-azurermps-maclinux?view=azurermps-6.11.0)
   
1. Connect Powershell to the Azure cluster and 
   (https://docs.microsoft.com/en-us/azure/azure-government/documentation-government-get-started-connect-with-ps)
1. Review and adjust parameters execute the deployment script at ./OrchestrationDriver/serviceDeploymentScript.ps1

1. While logged in Azure with powershell navigate to the OrchestrationDriver subfolder 
   of the project Github (), read 
  
   , copy and paste serviceDeploymentScript.ps1

generate a Azure File Storage Instance with the following 
   in structions and arguments (https://docs.microsoft.com/en-us/azure/storage/common/storage-powershell-guide-full)
    1.





and login (portal.azure.com 
or portal.azure.us respectively)
1. Install local Azure Powershell Instance (required on Gov't) to 
   (https://docs.microsoft.com/en-us/powershell/scripting/setup/installing-powershell-core-on-linux?view=powershell-6)
1. Setup a Service Principal to allow the Orchestration Driver to create and manipulate 
processing resources (https://docs.microsoft.com/en-us/azure/active-directory/develop/howto-create-service-principal-portal)
note ClientId, ClientSecret and TenantId for use in configuration later
1. Generate a Azure File Storage Instance (If using pre-generated test data)
1. Generate a CosmoDB Cassanda Instance
    1. Generate CosmoDB Cassandra API instance for the Python driver using the following instructions 
    (https://docs.microsoft.com/en-us/azure/cosmos-db/create-cassandra-python), providing 
    and noting a custom service name and endpoint url (which will be used for configuration later)*
    2. Navigate to the generated CosmoDB Cassandra resource on the portal and go to quickstart, 
    then note the CosmoDB secret (which will also be used for configuration)

### Local Setup
1. Ensure that the local environment has Internet access and can reach the portal.azure.us 
   domain via browser
1. Ensure that the local environment has a CLI accessible Docker, Git, Python 3.x and compatible pip 
   instance ('sudo apt install docker-compose, git, python3, pip3' on Ubuntu)
1. Ensure that the entire GitHub project repository is cloned locally 
   ('git clone https://github.com/booz-allen-hamilton/AzureMLSI2018.git')
1. From the root directory of the project, install the dependencies of the Orchestration 
   Driver 'pip3 install -r ./OrchestrationDriver/requirements.txt'
1. Copy and rename the "OrchestrationDriverConfigTemplate.json" to "OrchestrationDriverConfig.json"
   then populate with the Azure Storage and Azure CosmoDB Cassandra information noted 
   in the Azure setup instructions
1. Execute OrchestrationDriver.py with Python 3 'python3 ./OrchestrationDriver/OrchestrationDriver.py', 
   This should 
    1. Generate remaining required services in Azure including the VLAN, ContainerRegistry 
    and CognitiveServices
    1. Configure and build the other services in the repository, register them with 
    the Azure ContainerRegistry, and launch them

## Key Concepts 

* Programmatic Cloud Deployment

A key tenant of DevOps philosophy is the movement towards automation in development, 
testing, building and deployment of systems. By providing the ability to create and 
destroy virtually every key resource via API, public cloud services like Azure allow 
entire systems to be described and deployed by software with little to no human involvement.

Programmatic deployment exists on the opposite extreme of manual deployment using a 
GUI and requires a compatible DSL/Library and API

Programmatic deployment helps make complex systems more agile by making it easier to 
test the impact of changes at any level of the system

* Service Oriented Architecture

In this case services are simplified to mean REST services, meaning software systems 
which can only be interacted with via formally defined network interfaces (in this 
case HTTP based)

The advantage of the service approach is that it simplifies and formalizes communication 
between system components. This has wide ranging effects including the ability to break 
systems up into simple, easy to configure, easy to modify pieces. This, in turn, helps 
make otherwise complex undertakings like scaling, migrating, configuration change easier 
to execute.
 
Service architectures go in hand with Docker based containerization, as REST interfaces 
and ports are often the default means of communication in and out of a container, and 
a lot of products built atop docker address the complexities of orchestrating those 
communications across large numbers of both heterogeneous and homogenous services.

* Dynamic ML Retrain and Deployment
* Cloud Storage as a Service

One of the truths of cloud architecture is that data is the lowest common denominator 
of most information systems.
By this I mean that architecture should be dictated by the position and movement of 
data, not the other way round.
(and should be) more persistent 
then infrastructure. This means that it is usually advisable to separate data and storage 
from other types of less persistent.

This is represented in most Public Cloud architectures where multiple forms of long 
and middle term storage (ex. Amazon S3, Glacier) exist and typically become the  

Due in part to its relatively simple, linear key design the hardware footprint of Columnar 
NoSQL is relatively easy to extrapolate from its data volume (rows, indices, datatypes). 
This allows storage to be priced predictively based on data volume and access rates, rather then dynamic 
hardware based metrics like

## Demonstration System Services

The demonstration cluster consists of three Python web services which are built and distributed 
in Docker containers and deployed to an Azure cluster and the Orchestration Driver which 
builds the Azure cluster and builds/deploys the service containers. These services 
are scalable and can be replicated at will due to their reliance on Azure CosmosDB 
for storage.

### Orchestration Driver
The orchestration driver is a Python script which can be run on any Internet connected 
host which can reach the Azure API. Its two primary tasks are to: 
* Build and deploy the service containers
* Requisition and configure the shared services. 

More information can be found in the README.md inside its subfolder.

###Input Service Container
The input service is a Python web app which is run in a docker container which 

* Ingests prospective persona images into the CosmoDB to be processed by the TrainerClassifier 
  service and presented by the Visualization Services.
* Enriches these images with extracted faces using Cognitive Data Services prior to insertion into 
  CosmoDB 

Currently this data originates from the test data on Azure storage but may be replaced 
by a real time web crawler in the future.

More information can be found in the README.md inside its subfolder.

### Trainer Classifier Service Container
The Trainer Classifier Service Container is a Python web app which is run in a docker 
container which 

* Trains new versions of the facial classifier from training data stored 
in CosmoDB and generated through the Visualization Service
* Launches these new versions of the classifier via the AzureML API into containerized 
services
* Extracts unlabeled extracted faces and re-classifies them to determine their persona 
association then updates their status in CosmoDB

More information can be found in the README.md inside its subfolder.

### Visualization Service Container
The Visualization Service Container is a Python web app which is run in a docker container 
which

* Visualizes the personas current enumerated in CosmoDB
* Visualizes the raw images and extracted faces labeled to belong to each persona
* Visualizes the raw images and extracted faces predicted by the classifier to belong 
to each persona
* Labels predicted extracted faces as belonging or not belonging to a persona
* Inserting new personas into the CosmoDB database

More information can be found in the REAMDE.md inside its subfolder

## Azure Provided Components

### CosmoDB With Cassandra API
This DBaaS solution provides all of the storage and inter-service communication for the app. 

It stores the personas, raw images, extracted faces and associations used by the various 
parts of the system(table structure and fields are describe in greater detail in the Orchestration 
Driver README.md).

It also stores certain types of events (ex. labeling of images and addition personas) which allow  
services to monitor each others behavior and trigger behaviors in response (in this case 
retraining and deploying the classifier).

### Cognitive Services Facial Extractor
This is an out-of-the-box service provided by Azure, initialized by the Orchestration Driver and 
interacted with by the Input Service to extract faces from potential persona related 
images in order to be classified. 

The details of this system are discussed further 
in the README.md of the Input Service in its subfolder.