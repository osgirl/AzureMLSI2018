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

import yaml

import numpy as np
import cv2


import cognitive_face as CF

def getRawTestImagesFromBlobStore(bs_account_name, bs_account_key, bs_container):
    '''
        Retrieves the raw entity images from the configured azure blob store and returns in a list
    '''
    
    blob_service = BlockBlobService(account_name=bs_account_name, account_key=bs_account_key, endpoint_suffix="core.windows.net")
    logging.debug("Created blob service client using {0} account {1} container".format(bs_account_name, bs_container))

    blobs = blob_service.list_blobs(bs_container)
    logging.debug("Grabbed blob list in container {0}".format(bs_container));
    
    #Selects a sample of test data to test the parallelization of the ingest
    choice_blobs = random.choices(list(blobs), k=20)
    
    def byteGetter(blob):
        blob_bytes = blob_service.get_blob_to_bytes(bs_container, blob.name)
        return blob_bytes.content
    choice_blob_bytes = map(byteGetter, choice_blobs)
    
    return zip(choice_blobs, choice_blob_bytes)

def processEntityImages(choice_blobs, db_account_name, db_account_key, ca_file_uri, db_config, cs_account, cs_key):
    '''
    Tests entity images for the presence of a face using Azure Cognitive Services, extracts the face
    based on the provided bounding box, applies the facial classifier if available and then
    writes the raw image, face to CosmosDB
    '''

    #Initialize the cognitive services account to perform facial detection
    BASE_CS_URL = 'https://virginia.api.cognitive.microsoft.us/face/v1.0/'  # Replace with your regional Base URL
    CF.Key.set(cs_key)
    CF.BaseUrl.set(BASE_CS_URL)
    
    def extractFace(image_bytes):
        '''
        
        '''
        face_list = CF.face.detect(image_bytes)
        
        if len(face_list) == 1:
            face_rectangle = face_list[0]['faceRectangle']
            nparr = np.fromstring(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # cv2.IMREAD_COLOR in OpenCV 3.1
            face_iso_image = image[face_rectangle['top']:face_rectangle['top'] + face_rectangle['height'], 
                face_rectangle['left']:face_rectangle['left'] + face_rectangle['width']]
            return face_iso_image
        else:
            return None
    #Add the extracted face or none object to the end of the blob name and bytes tuple
    blob_image_faces = list(map(lambda blob_tuple: (blob_tuple[0], blob_tuple[1], extractFace(blob_tuple[1])), choice_blobs))

    #Connect to the database
    ssl_opts = {
        'ca_certs': ca_file_uri,
        'ssl_version': PROTOCOL_TLSv1_2,
        'cert_reqs': CERT_REQUIRED  # Certificates are required and validated
    }
    auth_provider = PlainTextAuthProvider(username=db_account_name, password=db_account_key)
    endpoint_uri = db_account_name + '.cassandra.cosmosdb.azure.com'
    cluster = Cluster([endpoint_uri], port = 10350, auth_provider=auth_provider, ssl_options=ssl_opts)

    #If no db config file is passed, look for the container environment variables
    if db_config is None:
        keyspace = os.environ['DB_KEYSPACE']
        personaTableName = os.environ['DB_PERSONA_TABLE']
        personaTableName = os.environ['DB_SUB_PERSONA_TABLE']
        personaEdgeTableName = os.environ['DB_SUB_PERSONA_EDGE_TABLE']
        rawImageTableName = os.environ['DB_RAW_IMAGE_TABLE']
        refinedImageTableName = os.environ['DB_REFINED_IMAGE_TABLE']
    #Otherwise load db config
    else:
        keyspace = db_config['db-keyspace']
        personaTableName = db_config['db-persona-table']
        subPersonaTableName = db_config['db-sub-persona-table']
        subPersonaEdgeTableName = db_config['db-sub-persona-edge-table']
        rawImageTableName = db_config['db-raw-image-table']
        refinedImageTableName = db_config['db-refined-image-table']

    #Prepare Cosmos DB session and insertion queries
    session = cluster.connect(keyspace);   
    personaInsertQuery = session.prepare("INSERT INTO " + personaTableName + " (persona_name) VALUES (?)")
    subPersonaInsertQuery = session.prepare("INSERT INTO " + subPersonaTableName + " (sub_persona_name, persona_name) VALUES (?, ?)")
    subPersonaEdgeInsertQuery = session.prepare("INSERT INTO " + subPersonaEdgeTableName + " (sub_persona_name, assoc_face_id, label_v_predict_assoc_flag) VALUES (?,?,?)")
    rawInsertQuery = session.prepare("INSERT INTO " + rawImageTableName + " (image_id, refined_image_edge_id, file_uri, image_bytes) VALUES (?,?,?,?)")
    refinedInsertQuery = session.prepare("INSERT INTO " + refinedImageTableName + " (image_id, raw_image_edge_id, image_bytes) VALUES (?,?,?)")
    
    for (blob, image_bytes, face_bytes) in blob_image_faces:
        if face_bytes is not None:
            file_name = blob.name
            (entity, usage, number) = file_name.split('-')
        
            #For each image extract the label and use to generate a persona, redundant writes will cancel out
            
            #Writes the entity label to the persona table 
            #For the time being also write the entity label to the subpersona table and associate with the persona table
            session.execute(personaInsertQuery, (entity,))
            session.execute(subPersonaInsertQuery, (entity, entity))
            
            #Writes the raw image to its table
            hashed_bytes = hashlib.md5(image_bytes).digest()
            hashed_bytes_int = int.from_bytes(hashed_bytes, byteorder='big') #Good identifier, sadly un
            image_hash = str(hashed_bytes_int) #Stupid workaround to Python high precision int incompatability
            #session.execute(rawInsertQuery, (image_hash, "", file_name, image_bytes))
            
            #Write each face extracted from the image to the DB as a refined image
            face_bytes = cv2.imencode(".jpg", face_bytes)[1]
            face_hash_bytes = hashlib.md5(face_bytes).digest()
            face_hashed_bytes_int = int.from_bytes(face_hash_bytes, byteorder='big') #Good identifier, sadly un
            face_hash = str(face_hashed_bytes_int) #Stupid workaround to Python high precision int incompatability
            session.execute(refinedInsertQuery, (face_hash, image_hash, face_bytes))
            logging.debug("Writing face to DB {0}".format(face_hash))
            
            #If the data is part of the training set, write edges between the sub-personas, face images
            if usage == "Train":
                session.execute(subPersonaEdgeInsertQuery, (entity, face_hash, True))
            #Otherwise do not write an edge, these will be predicted later by the training service

def main ():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    FORMAT = '%(asctime) %(message)s'
    logging.basicConfig(format=FORMAT)
    
    #Check if the blob storage access credentials have been loaded as a secret volume use them
    if os.path.exists('/tmp/secrets/bs/blob-storage-account'):
        bs_account = open('/tmp/secrets/bs/blob-storage-account').read()
        bs_key = open('/tmp/secrets/bs/blob-storage-key').read()
        logging.debug('Loaded db secrets from secret volume')

        db_account = open('/tmp/secrets/db/db-account').read()
        db_key = open('/tmp/secrets/db/db-key').read()
        logging.debug('Loaded db secrets from secret volume')
        
        #Load cognitive service credentials
        cs_account = open('/tmp/secrets/cs/cs-account').read()
        cs_key = open('/tmp/secrets/cs/cs-key').read()
        logging.debug('')

        
    #Otherwise assume it is being run locally and load from environment variables
    else: 
        bs_account = os.environ['BLOB_STORAGE_ACCOUNT']
        bs_key = os.environ['BLOB_STORAGE_KEY']
        
        db_account = os.environ['COSMOS_DB_ACCOUNT']
        db_key = os.environ['COSMOS_DB_KEY']
        
        #Load cognitive service credentials
        cs_account = os.environ['COG_SERV_ACCOUNT']
        cs_key = os.environ['COG_SERV_KEY']

    #If the test container is not loaded as an environment variable, assume a local run
    #and use the configuration information in the deployment config
    if 'BS_TEST_CON' in os.environ:
        bs_container_name = os.environ['BS_TEST_CON']
        db_config = None
    else:
        merged_config = yaml.safe_load_all(open("../cluster-deployment.yml"))
        for config in merged_config:
            print(config)
            if config['metadata']['name'] == 'azmlsi-bs-config':
                bs_container_name = config['data']['blob-storage-con']
            if config['metadata']['name'] == 'azmlsi-db-config':
                db_config  = config['data']
    
    ca_file_uri = "./cacert.pem"


    choice_blobs = getRawTestImagesFromBlobStore(bs_account, bs_key, bs_container_name)
    processEntityImages(choice_blobs, db_account, db_key, ca_file_uri, db_config, cs_account, cs_key)
    
if __name__ == '__main__':
    '''

    '''
    main()