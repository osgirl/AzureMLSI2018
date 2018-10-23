# Visualization Service Container

## Introduction

This service creates the graphic user interface (GUI) for the demonstration system 

## Instructions

This service must be started in both local testing and cloud deployment after the 
InitInputService and should be started after the TrainingClassificationService in order 
to have viewable input

##Local Testing
Run the InitInputService.py with the local Cassandra DB active, requirements.txt pip 
installed and a properly configured InitInputServiceConfig.py file involved.

##Remote Deployment
Build, register and deploy the containers

## Reference

The structure and docker components of this app are largely based on the following 
example
https://github.com/brennv/flask-app