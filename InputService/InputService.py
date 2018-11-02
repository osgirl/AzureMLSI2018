'''
    This script runs once upon initial instantiation of the container to load the test data.
    The container can be run without running this script. 
'''
from azure.storage.blob import BlockBlobService, PublicAccess
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from ssl import PROTOCOL_TLSv1_2, CERT_REQUIRED

import random
import os

import hashlib

import json

import logging
import sys

def testPopulateCassandraTables(session, cosmoCfg):
    '''
    Populates configured CosmoDB Cassandra Tables with test data taken from the configured Azure
    storage account
    '''
    personaTableName = cosmoCfg['personaTableName']
    personaEdgeTableName = cosmoCfg['personaEdgeTableName']
    rawImageTableName = cosmoCfg['rawImageTableName']
    refinedImageTableName = cosmoCfg['refinedImageTableName']

    #Iterates through labeled presidential images in a provided directory, extracts the label from 
    #the filename, and adds them as (randomly) labeled or unlabeled to the database
    personaInsertQuery = session.prepare("INSERT INTO " + personaTableName + " (persona_name) VALUES (?)")
    personaEdgeInsertQuery = session.prepare("INSERT INTO " + personaEdgeTableName + " (persona_name, assoc_image_id, label_assoc_flag, pred_assoc_flag) VALUES (?,?,?,?)")
    rawInsertQuery = session.prepare("INSERT INTO " + rawImageTableName + " (image_id, refined_image_edge_id, file_uri, image_bytes) VALUES (?,?,?,?)")
    refinedInsertQuery = session.prepare("INSERT INTO " + refinedImageTableName + " (image_id, raw_image_edge_id, image_bytes) VALUES (?,?,?)")

    for dirPath, dirNames, fileNames in os.walk("./TestImages"):
        for fileName in fileNames:
            print(fileName)
            fileHandle = open(dirPath + "/" + fileName, 'rb') #TODO make more robust with a non-OS specific seperator
            label = fileName.split("-")[0]
            choice = random.choice((0,1,2))
            
            #For each image extract the label and use to generate a persona, redundant writes will cancel out
            image_bytes = fileHandle.read();
            hashed_bytes = hashlib.md5(image_bytes).digest()
            hashed_bytes_int = int.from_bytes(hashed_bytes, byteorder='big') #Good identifier, sadly un
            hashed_bytes_str = str(hashed_bytes_int) #Stupid workaround to Python high precision int incompatability
            
            #
            session.execute(personaInsertQuery, (label,))
            session.execute(rawInsertQuery, (hashed_bytes_str, "", fileName, image_bytes))
            session.execute(refinedInsertQuery, (hashed_bytes_str, "", image_bytes))
            
            #Add image rows
            session.execute(refinedInsertQuery, (hashed_bytes_str, hashed_bytes_str, b""))
            
            #For one out of every three images register as label associated, otherwise predictive
            if choice == 2:
                session.execute(personaEdgeInsertQuery, (label, hashed_bytes_str, True, False))
            else:
                session.execute(personaEdgeInsertQuery, (label, hashed_bytes_str, False, True))
            
            #fileService.create_file_from_path(cosmoCfg.fileStorageShareName, cosmoCfg.fileStorageDir, fileName, dirPath + "/" + fileName)
            fileHandle.close()

def main ():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    FORMAT = '%(asctime) %(message)s'
    logging.basicConfig(format=FORMAT)
    
    #Load the Azure File storage credentials from the secret volume
    
    #Check if the blob storage access credentials have been loaded as a secret volume, then look at the environment variables for
    #testing
    if os.path.exists('/tmp/secrets/bs_account_name'):
        account_name = open('/tmp/secrets/bs/bs_account_name').read()
        account_key = open('/tmp/secrets/bs/bs_account_key').read()
        logging.debug('Loaded bs secrets from secret volume')
    else: 
        account_name = os.environ['AZ_BS_ACCOUNT_NAME']
        account_key = os.environ['AZ_BS_ACCOUNT_KEY']
        logging.debug('Loaded bs secrets from test environment variables')
    #Check if the blob storage access credentials have been loaded as a secret volume, then look at the environment variables for
    #testing
    #if os.path.exists('/tmp/secrets/db/db_account_name'):
    container_name = os.environ['AZ_BS_TEST_CON']
    
    blob_service = BlockBlobService(account_name=account_name, account_key=account_key, endpoint_suffix="core.usgovcloudapi.net")
    logging.debug("Created blob service client using {0} account {1} container".format(account_name, container_name))

    blobs = blob_service.list_blobs(container_name)
    logging.debug("Grabbed blob list in container {0}".format(container_name));
    
    #Selects a sample of test data to test the parallelization of the ingest
    choice_blobs = random.choices(list(blobs), k=20)
    
    #Generates additional traffic to the DB by repeatedly ingesting the selected sample of test data
    for x in range(0, 10):
        for blob in choice_blobs:
                blobBytes = blob_service.get_blob_to_bytes(container_name, blob.name)
                blob_bytes = logging.debug("Grabbed blob {0}".format(blob.name))
    
    
if __name__ == '__main__':
    '''


    '''
    main()
    
    '''
    cfg = json.load(open("InputServiceConfig.json", "r"))
    cosmoCfg = cfg['cosmoDBParams']
    if cosmoCfg['localDBEndpoint']:
        #Running the cluster from a local instance without security options engaged
        #https://www.digitalocean.com/community/tutorials/how-to-install-cassandra-and-run-a-single-node-cluster-on-ubuntu-14-04
        cluster = Cluster()
    else:
        #Cassandra connection options for the Azure CosmoDB with Cassandra API from the quickstart documentation on the portal page
        #grabbed my CA.pem from Mozilla https://curl.haxx.se/docs/caextract.html
        ssl_opts = {
            'ca_certs': './cacert.pem',
            'ssl_version': PROTOCOL_TLSv1_2,
            'cert_reqs': CERT_REQUIRED  # Certificates are required and validated
        }
        auth_provider = PlainTextAuthProvider(username=cosmoCfg['cosmoDBAccount'], password=cosmoCfg['cosmoDBSecret'])
        cluster = Cluster([cosmoCfg['cosmoDBEndpointUri']], port = 10350, auth_provider=auth_provider, ssl_options=ssl_opts)
        
    session = cluster.connect(cosmoCfg['cosmoDBKeyspaceName']);   

                
    testPopulateCassandraTables(session, cosmoCfg)
    
    persona_list = list(session.execute("SELECT persona_name FROM " + cosmoCfg['personaTableName']))
    
    for persona in persona_list:
        print(persona.persona_name)
    
    session.shutdown()
    '''