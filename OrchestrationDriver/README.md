# Intro
Orchestration driver which is build and executed outside of the Azure cluster in order 
to prepare the cluster for the deployment of the demonstration system.

# Details

1. Loads the Test Data (not stored in the Git repo, must be acquired and positioned locally) 
into the Azure File Storage service for later user by the components of the Demonstration 
cluster
2. Builds the Python packages of the other components
3. Builds the Docker containers of the other components, including the built Python 
   Packages
4. Deploys the component Docker containers to the Azure Container Storage system
5. Instantiates and configures the shared resources of the cluster
6. Instantiates the component Docker containers from the Azure Container Storage system 
   and configures them with the shared resources

# Instructions

pip3 install -r requirements.txt 
python3 OrchestrationDriver.py

# Resources

https://github.com/Azure-Samples/container-service-python-manage
https://docs.microsoft.com/en-us/azure/storage/files/storage-python-how-to-use-file-storage

### Training Data Set
The demonstration training data set will consist of a set of images and accompanying 
metadata which will be either used to build the classifier system, classified by that 
system, or in some cases both. This is because this data set will include an initial 
labeled set, as well as additional data which can be labeled after the initialization 
of the demonstration to enhance its performance.

Currently these images will consists of pictures of three individuals (presidents), 
potentially divided between "presidential" images (which are likely to be similar in 
age and composition and therefore easily classified), "non-presidential" images 
(more difficult to classify images of childhood, etc.) and various "noise" images.

The idea being to demonstrate how user supplied labeling after initialization allow 
the classifier to recognize previously unknown, dissimilar but still valid entities.

All of these images and metadata files will be stored as files linked by filename ('abc.jpg', 
'abc.json') on Azure File Storage where they can be accessed by the demonstration when 
required.

### Demonstration Orchestration Script

Python script downloaded from the demonstration GitHub with most of the other demonstration 
resources (apart from the Training Data) which carries out the preparatory activities 
required to launch the demonstration Azure cluster from outside the Azure environment.

1. Load the Training Data Set (Stored Separately) into Azure File API for the Demonstration 
   Initialization Script
2. Build the Azure Containers from the local demonstration components (Python and Docker files) 
   and submit to Azure for storage
3. Initialize and (where training data is not required) Configure the demonstration resources  
   including AD, VLAN, CosmoDB, Cognitive Services and Model Management accounts to 
   be interacted with by the components within the demonstration Azure Cluster  
4. Initialize the Demonstration Azure Cluster components from the stored containers 
   and pointing them to the requisite cluster resources

### Demonstration Initialization Script
The first containerized Demonstration Component running in the Azure cluster, consists 
of a Python script which runs once upon the initialization of the container, and so 
may share a container with the other, persistent services

The initialization script performs multiple activities leading up to the initial interaction 
of users with the system.

1. Pulling training data down from Azure File Storage and loading the labeled and unlabeled 
   sets into their respective State Repository (CosmoDB) tables 
2. Submitting the downloaded images to the Cognitive Services facial extractor, annotating 
   images which do not contain faces in the Cluster State Repository, and when faces are returned 
   appending those into the Cluster State Repository


