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
from azure.storage.blob import BlockBlobService, PublicAccess
#from azure.common.credentials import ServicePrincipalCredentials

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

import cv2
import numpy as np

from pathlib import Path

import argparse

import os
import logging
import sys

import json

import yaml

from azure.cosmos.cosmos_client import CosmosClient

def generateCosmoDBStructure(merged_config, db_name, db_key, ca_file_uri, db_config):
    '''
    Prepares the CosmoDB Cassandra instance for the demonstration by imposing the required keyspace and table structure
    '''
    
    db_config = db_config['data']
    db_keyspace = db_config['db-keyspace']
    
    endpoint_uri = db_name + '.cassandra.cosmosdb.azure.us'
    logging.debug("Connecting to CosmosDB Cassandra using {0} {1} {2} {3}".format(db_name, db_key, endpoint_uri, os.path.exists(ca_file_uri)))

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
    #Default keyspace settings from Azure CosmoDB
    session.execute('CREATE KEYSPACE IF NOT EXISTS ' + db_keyspace + ' WITH replication = {\'class\': \'NetworkTopologyStrategy\', \'datacenter\' : 1 }'); #Formatting is stupid on this string due to the additional curley braces
    
    session = cluster.connect(db_keyspace);   

    keyspace = db_config['db-keyspace']
    persona_table_name = db_config['db-persona-table']
    persona_edge_table_name = db_config['db-persona-edge-table']
    raw_image_table_name = db_config['db-raw-image-table']
    refined_image_table_name = db_config['db-refined-image-table']
    log_table_name = db_config['db-log-table']

    #Create table
    print("\nCreating Table")
    session.execute('DROP TABLE IF EXISTS ' + persona_table_name)
    session.execute('DROP TABLE IF EXISTS ' + persona_edge_table_name)
    session.execute('DROP TABLE IF EXISTS ' + raw_image_table_name)
    session.execute('DROP TABLE IF EXISTS ' + refined_image_table_name)
    session.execute('DROP TABLE IF EXISTS ' + log_table_name)

    #Persona table provides metadata on each persona in the demonstration database such as name and date of birth
    #This table is set up primarily to be scanned and used to pivot to the persona edge table to discover images 
    session.execute('CREATE TABLE IF NOT EXISTS ' + persona_table_name + ' (persona_name text, PRIMARY KEY(persona_name))');
    
    #Persona edge table contains the associations to pivot from a selected persona to its associated images
    #These associations can exist either due to explicit labeling or predicted labels
    session.execute('CREATE TABLE IF NOT EXISTS ' + persona_edge_table_name + ' (persona_name text, assoc_face_id text, label_v_predict_assoc_flag boolean, PRIMARY KEY(persona_name, assoc_face_id))');
    
    #Refined table stores extracted face blobs and associative edges to the raw image from which it was derived
    session.execute('CREATE TABLE IF NOT EXISTS ' + refined_image_table_name + ' (image_id text, raw_image_edge_id text, image_bytes blob, PRIMARY KEY(image_id))');
    
    #Raw table stores pre-extraction images that contain at least one face
    session.execute('CREATE TABLE IF NOT EXISTS ' + raw_image_table_name + ' (image_id text, refined_image_edge_id text, file_uri text, image_bytes blob, PRIMARY KEY(image_id))');
    
    #Log table which allows the services to track write operations where needed, indexed by timestamp to the hour resolution
    session.execute('CREATE TABLE IF NOT EXISTS ' + log_table_name + ' (event_timestamp timestamp, event_type text, event_text text, PRIMARY KEY(event_timestamp, event_type))');

def testLocalFaceDetect(source_dir):
    cnn_face_classifier = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")

    classifier_uri = "./Face_cascade.xml"
    logging.debug("Loading face cascade from {0} which exists {1}".format(classifier_uri, os.path.exists(classifier_uri)))
    cascade_face_classifier = cv2.CascadeClassifier(classifier_uri)

    federated_total_count = 0
    cnn_total_count = 0
    cascade_total_count = 0
    image_total_count = 0
    for dir_path, dir_names, file_names in os.walk(source_dir, topdown=True):
        for file_name in file_names:
            image_uri = os.path.join(dir_path, file_name)
            print(image_uri)
            image=cv2.imread(image_uri)
            
            image_grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            faces = cascade_face_classifier.detectMultiScale(image_grey,scaleFactor=1.16,minNeighbors=5,minSize=(25,25),flags=0) 
            
            (h, w) = image.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(image, (300,300)), 1.0, (300, 300), (103.93, 116.77, 123.68))
            cnn_face_classifier.setInput(blob)
            detections = cnn_face_classifier.forward()
            faceCount = 0
            for i in range(0, detections.shape[2]):                
                confidence = detections[0, 0, i, 2]
                if confidence > 0.95:
                    #print("{0} {1}".format(i, confidence))
                    faceCount += 1
                #box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                #(startX, startY, endX, endY) = box.astype("int") 
                #print("{0} {1} {2} {3}".format(startX, startY, endX, endY))
            
                
            if len(faces) > 0:
                federatedResult = len(faces)
            elif faceCount > 0:
                federatedResult = 1
            else:
                federatedResult = 0
            print("Found {0} and {1} and {2} faces".format(len(faces), faceCount, federatedResult))
            cascade_total_count += len(faces)
            cnn_total_count += faceCount
            federated_total_count += federatedResult
            image_total_count += 1
            
            
            
    print("{0} {1} {2} {3} {4}".format(image_total_count, federated_total_count, cascade_total_count/image_total_count, cnn_total_count/image_total_count, federated_total_count/image_total_count))

def generateAzureInputStore(bs_config, stor_name, stor_key, source_dir):
    '''
    Loads a folder of images with the appropriate filenames into the Azure Blob Storage dir so they are accessible to Input
    workers running in the cloud
    '''
        
    bs_dir_name = bs_config['data']['az-bs-test-dir']
    
    block_blob_service = BlockBlobService(account_name=stor_name, account_key=stor_key, endpoint_suffix="core.usgovcloudapi.net")
    block_blob_service.create_container(bs_dir_name)
    logging.debug("Connected to blob service {0}".format(stor_name))

    image_count = 0
    for dir_path, dir_names, file_names in os.walk(source_dir, topdown=True):
        for file_name in file_names:
            dir_components = Path(dir_path)
            print(dir_components)
            usage = dir_components.parts[len(dir_components.parts) - 1]
            entity = dir_components.parts[len(dir_components.parts) - 2]
            blob_name = usage + "-" + entity + "-" + str(image_count)
            block_blob_service.create_blob_from_path(bs_dir_name, blob_name, dir_path + "/" + file_name)
            logging.debug("File written to blob container {0} from {1} {2}".format(bs_dir_name, os.path.join(dir_path, file_name), blob_name))
            image_count += 1

def main():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    FORMAT = '%(asctime) %(message)s'
    logging.basicConfig(format=FORMAT)
    
    #Check if the blob storage access credentials have been loaded as a secret volume, then look at the environment variables for
    #testing
    
    db_account_name = os.environ['DB_ACCOUNT']
    db_account_key = os.environ['DB_KEY']
    
    bs_account_name = os.environ['BLOB_STORAGE_ACCOUNT']
    bs_account_key = os.environ['BLOB_STORAGE_KEY']

    merged_config_uri = os.environ['CONFIG_URI']
    merged_config = yaml.safe_load_all(open(merged_config_uri))

    for config in merged_config:
        if config['metadata']['name'] == 'azmlsi-db-config':
            db_config = config
        if config['metadata']['name'] == 'azmlsi-bs-config':
            bs_config = config

    #cfg = json.load(open('./OrchestrationDriverConfig.json', 'r'))
    ca_file_uri = "./cacert.pem"
    source_dir = "./TestImages"
    
    generateAzureInputStore(bs_config, bs_account_name, bs_account_key, source_dir)
    generateCosmoDBStructure(config, db_account_name, db_account_key, ca_file_uri, db_config)

    

if __name__ == '__main__':
    main()