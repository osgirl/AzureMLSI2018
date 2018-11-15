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
import cv2
import numpy as np

#Loads the Python face detection classifiers from their local serialized forms
cnn_face_classifier = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")
classifier_uri = "./Face_cascade.xml"
cascade_face_classifier = cv2.CascadeClassifier(classifier_uri)

def detectExtractFace(image_bytes):
    #Run the Cascade face detector
    nparr = np.fromstring(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # cv2.IMREAD_COLOR in OpenCV 3.1
    image_grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = cascade_face_classifier.detectMultiScale(image_grey,scaleFactor=1.16,minNeighbors=5,minSize=(25,25),flags=0) 
    
    #Run the CNN face detector
    (h, w) = image.shape[:2] #Save off original image dimensions
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300,300)), 1.0, (300, 300), (103.93, 116.77, 123.68))
    cnn_face_classifier.setInput(blob)
    detections = cnn_face_classifier.forward()
    
    #If Cascade identifies at least one face, use the cascade results to extract the faces
    roi_images = []
    federatedResult = 0
    if len(faces) > 0:
        federated_result = 2
        for (x,y,w,h) in faces:
            roi_image = image[y:(y+h), x:(x+w)]
            roi_images.append(roi_image)
    #Otherwise
    else:
        #Grab only first element from the CNN classifier
        element = 1
        confidence = detections[0,0,element,2]
        if confidence > 0.95:
            federatedResult = 1
            box = detections[0, 0, element, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            roi_image = image[startY, endY, startX, endX]
            roi_images.append(roi_image)
        
    '''
    faceCount = 0
    for i in range(0, detections.shape[2]):                
        confidence = detections[0, 0, i, 2]
        if confidence > 0.95:
            #print("{0} {1}".format(i, confidence))
            faceCount += 1
        federatedResult = len(faces)
    print("Found {0} and {1} and {2} faces".format(len(faces), faceCount, federatedResult))
    '''
    counter = 0
    for face in roi_images:
        logging.debug("Face found!")
        cv2.imwrite("face" + str(counter) + ".jpg", face)
            
    return roi_images

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

def testPopulateCassandraTables(choice_blobs, db_account_name, db_account_key, ca_file_uri, db_config):
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

    #If no db config file is passed, look for the container environment variables
    if db_config is None:
        keyspace = os.environ['DB_KEYSPACE']
        personaTableName = os.environ['DB_PERSONA_TABLE']
        personaEdgeTableName = os.environ['DB_PERSONA_EDGE_TABLE']
        rawImageTableName = os.environ['DB_RAW_IMAGE_TABLE']
        refinedImageTableName = os.environ['DB_REFINED_IMAGE_TABLE']
    #Otherwise load db config
    else:
        keyspace = db_config['db-keyspace']
        personaTableName = db_config['db-persona-table']
        personaEdgeTableName = db_config['db-persona-edge-table']
        rawImageTableName = db_config['db-raw-image-table']
        refinedImageTableName = db_config['db-refined-image-table']

    session = cluster.connect(keyspace);   

    #Iterates through labeled presidential images in a provided directory, extracts the label from 
    #the filename, and adds them as (randomly) labeled or unlabeled to the database
    personaInsertQuery = session.prepare("INSERT INTO " + personaTableName + " (persona_name) VALUES (?)")
    personaEdgeInsertQuery = session.prepare("INSERT INTO " + personaEdgeTableName + " (persona_name, assoc_image_id, label_v_predict_assoc_flag) VALUES (?,?,?,?)")
    rawInsertQuery = session.prepare("INSERT INTO " + rawImageTableName + " (image_id, refined_image_edge_id, file_uri, image_bytes) VALUES (?,?,?,?)")
    refinedInsertQuery = session.prepare("INSERT INTO " + refinedImageTableName + " (image_id, raw_image_edge_id, image_bytes) VALUES (?,?,?)")
    
    
    for (blob, blob_bytes) in choice_blobs:
        file_name = blob.name
        (entity, usage, number) = file_name.split('-')
        logging.debug("Writing blob name {0} into the table".format(file_name))
        
        #For each image extract the label and use to generate a persona, redundant writes will cancel out
        image_bytes = blob_bytes;
        face_list = detectExtractFace(image_bytes)
        
        #Write only if faces are found
        if len(face_list) > 0:
            hashed_bytes = hashlib.md5(image_bytes).digest()
            hashed_bytes_int = int.from_bytes(hashed_bytes, byteorder='big') #Good identifier, sadly un
            image_hash = str(hashed_bytes_int) #Stupid workaround to Python high precision int incompatability
            
            #Writes the entity label to the DB, will overwrite for identical labels
            session.execute(personaInsertQuery, (entity,))
            
            #Writes an individual image to the DB
            session.execute(rawInsertQuery, (image_hash, "", file_name, image_bytes))
            
            #Write each face extracted from the image to the DB as a refined image
            for face_bytes in face_list:
                face_bytes = cv2.imencode(".jpg", face_bytes)[1]
                face_hash_bytes = hashlib.md5(face_bytes).digest()
                face_hashed_bytes_int = int.from_bytes(face_hash_bytes, byteorder='big') #Good identifier, sadly un
                face_hash = str(face_hashed_bytes_int) #Stupid workaround to Python high precision int incompatability
                session.execute(refinedInsertQuery, (face_hash, image_hash, face_bytes))
            
            #Writes an associative edge between the entity and raw and refined images that come from labeled training data
            #Usage is conveyed from the filename that was used to load the image into cluster storage
            if usage == "Train":
                session.execute(personaEdgeInsertQuery, (entity, image_hash, True))
                session.execute(personaEdgeInsertQuery, (entity, face_hash, True))

def main ():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    FORMAT = '%(asctime) %(message)s'
    logging.basicConfig(format=FORMAT)
    
    #Check if the blob storage access credentials have been loaded as a secret volume use them
    if os.path.exists('/tmp/secrets/bs/blob-storage-account'):
        bs_account_name = open('/tmp/secrets/bs/blob-storage-account').read()
        bs_account_key = open('/tmp/secrets/bs/blob-storage-key').read()
        logging.debug('Loaded db secrets from secret volume')

        db_account_name = open('/tmp/secrets/db/db-account').read()
        db_account_key = open('/tmp/secrets/db/db-key').read()
        logging.debug('Loaded db secrets from secret volume')
    #Otherwise assume it is being run locally and load from environment variables
    else: 
        bs_account_name = os.environ['AZ_BS_ACCOUNT_NAME']
        bs_account_key = os.environ['AZ_BS_ACCOUNT_KEY']
        logging.debug('Loaded db secrets from test environment variables')
        
        db_account_name = os.environ['AZ_DB_ACCOUNT_NAME']
        db_account_key = os.environ['AZ_DB_ACCOUNT_KEY']

    #If the test container is not loaded as an environment variable, assume a local run
    #and use the configuration information in the deployment config
    if 'BS_TEST_CON' in os.environ:
        bs_container = os.environ['BS_TEST_CON']
        db_config = None
    else:
        merged_config = yaml.safe_load_all(open("../cluster-deployment.yml"))
        for config in merged_config:
            print(config)
            if config['metadata']['name'] == 'azmlsi-bs-config':
                bs_test_container = config['data']['az-bs-test-dir']
            if config['metadata']['name'] == 'azmlsi-db-config':
                db_config  = config['data']
    
    ca_file_uri = "./cacert.pem"

    choice_blobs = retrieveTestImageSample(bs_account_name, bs_account_key, bs_test_container)
    testPopulateCassandraTables(choice_blobs, db_account_name, db_account_key, ca_file_uri, db_config)
    
if __name__ == '__main__':
    '''

    '''
    main()