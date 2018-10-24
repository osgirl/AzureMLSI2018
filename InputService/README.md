# Initialization/Ingest Script

## Introduction
This container is intended to one-time initalize the cluster by loading the demonstration training 
data from Azure storage, loading it into a local or Azure (CosmoDB) hosted Cassandra compatible 
database, enriching it with faces extracted using the Azure Cognative Services API 
and, potentially.

Eventually it may also create a continuously running service to to crawl for additional images to 
be inserted into the Cassandra database.

Note that this service does not interact directly with the retraining or reclassification 
service except through publishing training data and retraining events to the Cassandra 
database which triggers training events.
