# AzureMLSI2018
##Introduction
Awesome project experimenting with assembling a realm time ML processing system using 
Python, Azure API, Azure Model Management Services, Azure Cognitive Services, Azure
File Storage and CosmoDB.

##Custom System Components
###Training Data Set
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

###Demonstration Orchestration Script

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

###Demonstration Initialization Script
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



###Model Management Services Facial Classifier Trainer
This is a Python/C++ script running on either a Container (assuming they can be GPU 
accelerated) or a VM. This service carries out two basic actions: retraining the facial 
classifier and batch relabeling.

####Facial Classifier Retraining
When a new classifier is produced by the retraining process, this component will extract 
all the faces for non-labeled images in CosmoDB and re-predict their labels based on 
the new classifier and update with these new predicted labels 

####Batch Relabeling
Upon updating the classifier, this service may also re-run unlabeled faces stored in the Cluster 
State Repository against the classifier and update their predicted labels for the display.

### Demonstration User Interface
Python web server (Flask??) and HTML/JS frontend (Reactive.js??) which provides the web accessible user interface 
for the demonstration system. Primarily allows users to 

1. View Personas available in the current collected image set
2. View Labeled and Predicted Label Images Assigned to Each Persona (Note this step 
   will activate the facial classifier trainer to retrain and relabel the image store)
3. Confirm Predicted Labeled Images to add them to the Labeled set for training 
4. Upload new images (either directly or via provided webpages to be scraped) to the current 
   collected image set for feature extraction and labeling (Note this step will work 
   with the Cognition Services and Model Management services to extract facial features 
   and classify the new provided images)
5. Monitor the current status of the classifier through metadata provided by the model 
   management API

## Azure Provided Components

###Cognitive Services Facial Extractor
This is an out-of-the-box service provided by Azure (not a containerized Python script), 
initialized by the orchestration script and used by the demonstration initialization 
script to determine if faces exist in test images and extract those faces for classification

###System State Repository (CosmoDB)