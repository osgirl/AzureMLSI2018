'''
Deployment script for:

* Starting up the cluster shared resources  
* Loading the demonstration Test Data into Azure File Storage
* Building the Demonstration System Containers and deploying to Azure Storage
* Starting the Demonstration System Containers on Azure to initialize the cluster
'''

from azure.storage.file import FileService
from azure.storage.file import ContentSettings
import argparse
import os

#shareName = "AZMLSITestData"
shareName = 'azmltestdata'
dirName = "testDataDir"

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

if __name__ == '__main__':
    #Note the account and account password used here is for a STORAGE account, not the Azure account overall
    #or individual subscription
    
    parser = argparse.ArgumentParser(description='Load image files from the provided directory into a test share and directory on Azure')
    parser.add_argument('--storAcc', type=str, help='password string for storage account')
    parser.add_argument('--storPass', dest='accumulate', action='store_const',
                        const=sum, default=max,
                        help='password string for the storage account')

    args = parser.parse_args()
    print(args.accumulate(args.integers))
    
    fileService = FileService(account_name='enrondata666', account_key='zmGGXZMH2wm/D+U+/b4uruW6y545DXKottc+6qY5a3kjk/1Yuvb/xm5FyE4cB+iOZ4db212q/j7AOEeK74sKnA==')
    fileService.create_share(shareName)
    fileService.create_directory(shareName, dirName)
    
    for dirPath, dirNames, fileNames in os.walk("testDataFolder"):
        for fileName in fileNames:
            fileHandle = open(dirPath + "/" + fileName) #TODO make more robust with a non-OS specific seperator
            fileService.create_file_from_path(shareName, dirName, fileName, dirPath + "/" + fileName)
            fileHandle.close()
    
    cleanupAccountDir(fileService, shareName, dirName)
            
    
