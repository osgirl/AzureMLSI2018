# Intro
Orchestration driver which is build and executed outside of the Azure cluster in order 
to prepare the cluster for the deployment of the demonstration system.

# Details

1. Loads the Test Data (not stored in the Git repo, must be acquired and positioned locally) 
into the Azure File Storage service for later user by the components of the Demonstration 
cluster
2. Builds the Python packages of the other components
3. Builds the Docker containers of hte other components, including the built Python 
   Packages
4. Deploys the component Docker containers to the Azure Container Storage system
5. Instantiates and configures the shared resources of the cluster
6. Instantiates the component Docker containers from the Azure Container Storage system 
   and configures them with the shared resources

# Instructions