# Retrainer/Reclassifier Script

## Introduction
This is a containerized Python service which takes faces extracted from labeled portraits 
stored in CosmoDB/Cassandra, uses a trained VGG CNN to generate a visual feature vector, 
then classifies those images based on a Random Forest Classifier with a predicted label, then 
apprends that predicted label to the Cassandra record

## Components
###CosmoDB/Cassandra
Both the inputs and outputs of the retraining

###Oxford Visual Geometry Group (VGG) Convolutional Neural Network (CNN) 
 