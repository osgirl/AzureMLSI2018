# Initialization/Ingest Script

## Introduction
This container is intended to initalize the cluster by loading the training data from Azure storage, 
loading it into CosmoDB and enriching it with faces extracted from the Cognitive Services API. 

It may also eventually include an active service with crawls for additional web images 
to run against the classifier and display in the interface