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

def retrieveTestImageSample(bs_account_name, bs_account_key, bs_container):
    '''

    '''
    blob_service = BlockBlobService(account_name=bs_account_name, account_key=bs_account_key, endpoint_suffix="core.usgovcloudapi.net")
    logging.debug("Created blob service client using {0} account {1} container".format(bs_account_name, bs_container))

    blobs = blob_service.list_blobs(bs_container)
    logging.debug("Grabbed blob list in container {0}".format(bs_container));
    
    #Selects a sample of test data to test the parallelization of the ingest
    choice_blobs = random.choices(list(blobs), k=20)
    
    def byteGetter(blob):
        blob_bytes = blob_service.get_blob_to_bytes(bs_container, blob.name)
        print(blob_bytes.content)
        return blob_bytes.content
    choice_blob_bytes = map(byteGetter, choice_blobs)
    
    return zip(choice_blobs, choice_blob_bytes)

def testPopulateCassandraTables(choice_blobs, db_account_name, db_account_key, ca_file_uri):
    '''
    Populates configured CosmoDB Cassandra Tables with test data taken from the configured Azure
    storage account
    '''

    ssl_opts = {
        'ca_certs': ca_file_uri,
        'ssl_version': PROTOCOL_TLSv1_2,
        'cert_reqs': CERT_REQUIRED  # Certificates are required and validated
    }
    auth_provider = PlainTextAuthProvider(username=db_account_name, password=db_account_key)
    endpoint_uri = db_account_name + '.cassandra.cosmosdb.azure.us'
    cluster = Cluster([endpoint_uri], port = 10350, auth_provider=auth_provider, ssl_options=ssl_opts)
    
    

    personaTableName = os.environ['AZ_DB_PERSONA_TABLE']
    personaEdgeTableName = os.environ['AZ_DB_PERSONA_EDGE_TABLE']
    rawImageTableName = os.environ['AZ_DB_RAW_IMAGE_TABLE']
    refinedImageTableName = os.environ['AZ_DB_REFINED_IMAGE_TABLE']

    '''
    cosmoCfg = cosmoCfg['']
    personaTableName = cosmoCfg['personaTableName']
    personaEdgeTableName = cosmoCfg['personaEdgeTableName']
    rawImageTableName = cosmoCfg['rawImageTableName']
    refinedImageTableName = cosmoCfg['refinedImageTableName']
    '''
    session = cluster.connect(os.environ['AZ_DB_KEYSPACE']);   

    #Iterates through labeled presidential images in a provided directory, extracts the label from 
    #the filename, and adds them as (randomly) labeled or unlabeled to the database
    personaInsertQuery = session.prepare("INSERT INTO " + personaTableName + " (persona_name) VALUES (?)")
    personaEdgeInsertQuery = session.prepare("INSERT INTO " + personaEdgeTableName + " (persona_name, assoc_image_id, label_assoc_flag, pred_assoc_flag) VALUES (?,?,?,?)")
    rawInsertQuery = session.prepare("INSERT INTO " + rawImageTableName + " (image_id, refined_image_edge_id, file_uri, image_bytes) VALUES (?,?,?,?)")
    refinedInsertQuery = session.prepare("INSERT INTO " + refinedImageTableName + " (image_id, raw_image_edge_id, image_bytes) VALUES (?,?,?)")
    
    for (blob, blob_bytes) in choice_blobs:
        file_name = blob.name
        logging.debug("Writing blob name {0} into the table".format(file_name))
        label = file_name.split("-")[0]
        choice = random.choice((0,1,2))
        
        #For each image extract the label and use to generate a persona, redundant writes will cancel out
        print(blob_bytes)
        image_bytes = blob_bytes;
        hashed_bytes = hashlib.md5(image_bytes).digest()
        hashed_bytes_int = int.from_bytes(hashed_bytes, byteorder='big') #Good identifier, sadly un
        hashed_bytes_str = str(hashed_bytes_int) #Stupid workaround to Python high precision int incompatability
        
        #
        session.execute(personaInsertQuery, (label,))
        session.execute(rawInsertQuery, (hashed_bytes_str, "", file_name, image_bytes))
        session.execute(refinedInsertQuery, (hashed_bytes_str, "", image_bytes))
        
        #Add image rows
        session.execute(refinedInsertQuery, (hashed_bytes_str, hashed_bytes_str, b""))
        
        #For one out of every three images register as label associated, otherwise predictive
        if choice == 2:
            session.execute(personaEdgeInsertQuery, (label, hashed_bytes_str, True, False))
        else:
            session.execute(personaEdgeInsertQuery, (label, hashed_bytes_str, False, True))
        

def main ():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    FORMAT = '%(asctime) %(message)s'
    logging.basicConfig(format=FORMAT)
    
    #Check if the blob storage access credentials have been loaded as a secret volume, then look at the environment variables for
    #testing
    if os.path.exists('/tmp/secrets/bs_account_name'):
        bs_account_name = open('/tmp/secrets/bs/bs_account_name').read()
        bs_account_key = open('/tmp/secrets/bs/bs_account_key').read()
        logging.debug('Loaded db secrets from secret volume')
    else: 
        bs_account_name = os.environ['AZ_BS_ACCOUNT_NAME']
        bs_account_key = os.environ['AZ_BS_ACCOUNT_KEY']
        logging.debug('Loaded db secrets from test environment variables')

    if os.path.exists('/tmp/secrets/db/db_account_name'):
        db_account_name = open('/tmp/secrets/db/db_account_name').read()
        db_account_key = open('/tmp/secrets/db/db_account_key').read()
        logging.debug('Loaded db secrets from secret volume')
    else:
        db_account_name = os.environ['AZ_DB_ACCOUNT_NAME']
        db_account_key = os.environ['AZ_DB_ACCOUNT_KEY']
    bs_test_container = os.environ['AZ_BS_TEST_CON']
    
    ca_file_uri = "./cacert.pem"

    #Check if the blob storage access credentials have been loaded as a secret volume, then look at the environment variables for
    #testing
    #if os.path.exists('/tmp/secrets/db/db_account_name'):
    container_name = os.environ['AZ_BS_TEST_CON']
    choice_blobs = retrieveTestImageSample(bs_account_name, bs_account_key, bs_test_container)
    testPopulateCassandraTables(choice_blobs, db_account_name, db_account_key, ca_file_uri)
    
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