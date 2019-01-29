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
from azure.storage.blob import BlockBlobService

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from ssl import PROTOCOL_TLSv1_2, CERT_REQUIRED

import numpy as np

from pathlib import Path
import hashlib

import os
import logging
import sys

import json
import yaml


def generateCosmoDBStructure(merged_config, db_name, db_key, ca_file_uri, db_config):
    '''
    Prepares the CosmoDB Cassandra instance for the demonstration by imposing the required keyspace and table structure
    '''
    
    db_config = db_config['data']
    db_keyspace = db_config['db-keyspace']
    
    endpoint_uri = db_name + '.cassandra.cosmosdb.azure.com'
    #endpoint_uri = db_name + '.cassandra.cosmosdb.azure.us'

    logging.debug("Connecting to CosmosDB Cassandra using {0} {1} {2}".format(db_name, endpoint_uri, os.path.exists(ca_file_uri)))

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
    print("\nCreating Keyspace {0}".format(db_keyspace))
    session = cluster.connect()
    response = session.execute('CREATE KEYSPACE IF NOT EXISTS ' + db_keyspace + ' WITH replication = {\'class\': \'NetworkTopologyStrategy\', \'datacenter\' : 1 }'); #Formatting is stupid on this string due to the additional curley braces
    session.shutdown()
    
    session = cluster.connect(db_keyspace);   

    keyspace = db_config['db-keyspace']
    persona_table_name = db_config['db-persona-table']
    sub_persona_table_name = db_config['db-sub-persona-table']
    sub_persona_face_edge_table_name = db_config['db-sub-persona-face-edge-table']
    face_sub_persona_edge_table_name = db_config['db-face-sub-persona-edge-table']
    raw_image_table_name = db_config['db-raw-image-table']
    face_image_table_name = db_config['db-face-image-table']
    log_offset_table_name = db_config['db-log-offset-table']

    #Create table
    print("\nCreating Table")
    session.execute('DROP TABLE IF EXISTS ' + persona_table_name)
    session.execute('DROP TABLE IF EXISTS ' + sub_persona_table_name)
    session.execute('DROP TABLE IF EXISTS ' + sub_persona_face_edge_table_name)
    session.execute('DROP TABLE IF EXISTS ' + face_sub_persona_edge_table_name)
    session.execute('DROP TABLE IF EXISTS ' + raw_image_table_name)
    session.execute('DROP TABLE IF EXISTS ' + face_image_table_name)
    session.execute('DROP TABLE IF EXISTS ' + log_offset_table_name)

    #Persona table provides metadata on each persona in the demonstration database such as name and date of birth
    #This table is set up primarily to be scanned and used to pivot to the persona edge table to discover images 
    session.execute('CREATE TABLE IF NOT EXISTS ' + persona_table_name + ' (persona_name text, PRIMARY KEY(persona_name))');
    
    #
    session.execute('CREATE TABLE IF NOT EXISTS ' + sub_persona_table_name + ' (sub_persona_name text, persona_name text, PRIMARY KEY(persona_name, sub_persona_name))');
    
    #Two tables which prove edges to associate sub-personas and face-images using labels or predictions and allow them to be discovered from either direction
    session.execute('CREATE TABLE IF NOT EXISTS ' + sub_persona_face_edge_table_name + ' (sub_persona_name text, assoc_face_id text, label_v_predict_assoc_flag boolean, PRIMARY KEY(sub_persona_name, assoc_face_id))');
    session.execute('CREATE TABLE IF NOT EXISTS ' + face_sub_persona_edge_table_name + ' (sub_persona_name text, assoc_face_id text, label_v_predict_assoc_flag boolean, PRIMARY KEY(assoc_face_id, sub_persona_name))');
    
    #Refined table stores extracted face blobs and associative edges to the raw image from which it was derived
    session.execute('CREATE TABLE IF NOT EXISTS ' + face_image_table_name + ' (face_id text, raw_image_edge_id text, face_bytes blob, feature_bytes blob, PRIMARY KEY(face_id))');
 
    #Raw table stores pre-extraction images that contain at least one face
    session.execute('CREATE TABLE IF NOT EXISTS ' + raw_image_table_name + ' (image_id text, file_uri text, image_bytes blob, PRIMARY KEY(image_id))');
    
    #Log table which allows the services to track write operations where needed, indexed by timestamp to the hour resolution
    session.execute('CREATE TABLE IF NOT EXISTS ' + log_offset_table_name + ' (event_timestamp timestamp, current_offset bigint, PRIMARY KEY(event_timestamp))');
    session.shutdown()

def generateAzureInputStore(bs_config, stor_name, stor_key, source_dir):
    '''
    Loads a folder of images with the appropriate filenames into the Azure Blob Storage dir so they are accessible to Input
    workers running in the cloud
    '''
        
    bs_dir_name = bs_config['data']['blob-storage-con']
    
    storage_uri=""
    block_blob_service = BlockBlobService(account_name=stor_name, account_key=stor_key)
                                          #, endpoint_suffix="core.usgovcloudapi.net") #addendum for use on gov't
    block_blob_service.create_container(bs_dir_name)
    logging.debug("Connected to blob service {0}".format(stor_name))

    image_count = 0
    for dir_path, dir_names, file_names in os.walk(source_dir, topdown=True):
        for file_name in file_names:
            dir_components = Path(dir_path)
            usage = dir_components.parts[len(dir_components.parts) - 1]
            entity = dir_components.parts[len(dir_components.parts) - 2]
            
            image_file = open(os.path.join(dir_path, file_name), 'rb').read()
            
            #Calculates image hash, infers the purpose of an image from its folder position and generate a filename
            hash = hashlib.md5(image_file).hexdigest()    
            blob_name = usage + "-" + entity + "-" + str(hash)
            
            #Uploads to the Azure Blob Store
            block_blob_service.create_blob_from_path(bs_dir_name, blob_name, dir_path + "/" + file_name)
            logging.debug("File written to blob container {0} from {1} {2}".format(bs_dir_name, os.path.join(dir_path, file_name), blob_name))
            
            #Renames the image in place
            os.rename(os.path.join(dir_path, file_name), os.path.join(dir_path, blob_name))
            logging.debug("Renamed {0} to {1}".format(os.path.join(dir_path, file_name), os.path.join(dir_path, blob_name)))

def main():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    FORMAT = '%(asctime) %(message)s'
    logging.basicConfig(format=FORMAT)
    
    #Check if the blob storage access credentials have been loaded as a secret volume, then look at the environment variables for
    #testing
    
    db_account_name = os.environ['COSMOS_DB_ACCOUNT']
    db_account_key = os.environ['COSMOS_DB_KEY']
    
    bs_account_name = os.environ['BLOB_STORAGE_ACCOUNT']
    bs_account_key = os.environ['BLOB_STORAGE_KEY']

    #merged_config_uri = "../cluster-deployment.yml"
    merged_config_uri = os.environ['CLUSTER_CONFIG_URI']
    merged_config = yaml.safe_load_all(open(merged_config_uri))
    for config in merged_config:
        if config['metadata']['name'] == 'azmlsi-db-config':
            db_config = config
        if config['metadata']['name'] == 'azmlsi-bs-config':
            bs_config = config
            bs_container = config['data']['blob-storage-con']

    #cfg = json.load(open('./OrchestrationDriverConfig.json', 'r'))
    #ca_file_uri = "./cacert.pem"
    ca_file_uri = os.environ["CA_FILE_URI"]
    source_dir = "./TestImages"
    
    generateAzureInputStore(bs_config, bs_account_name, bs_account_key, source_dir)
    generateCosmoDBStructure(config, db_account_name, db_account_key, ca_file_uri, db_config)

    

if __name__ == '__main__':
    main()