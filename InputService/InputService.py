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

import datetime

import yaml
import io

import numpy as np
#import cv2

import cognitive_face as CF
from PIL import Image

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

from keras import backend as K
from keras.applications import VGG19
from keras_vggface.vggface import VGGFace

from keras.applications import imagenet_utils

import avro.schema
import avro.io

import math
from azure.mgmt.eventhub.models import eh_namespace

img_width = 300
img_height = 300
vgg_face_feature_gen = VGGFace(include_top=False, input_shape=(img_width, img_height, 3), pooling='avg') # pooling: None, avg or max
schema = avro.schema.Parse(open("./VGGFaceFeatures.avsc", "rb").read())

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

from keras import backend as K
from keras import applications
from keras_vggface.vggface import VGGFace


from sklearn import tree
import warnings
import pickle

from azure.eventhub import EventHubClient, Sender, EventData

def getRawTestImagesFromBlobStore(bs_account_name, bs_account_key, bs_container):
    '''
        Retrieves the raw entity images from the configured azure blob store and returns in a list
    '''
    
    blob_service = BlockBlobService(account_name=bs_account_name, account_key=bs_account_key, endpoint_suffix="core.windows.net")
    logging.info("Created blob service client using {0} account {1} container".format(bs_account_name, bs_container))

    choice_blobs = blob_service.list_blobs(bs_container)
    logging.info("Grabbed blob list in container {0}".format(bs_container));
    
    #Selects a sample of test data to test the parallelization of the ingest
    #choice_blobs = random.choices(list(blobs), k=20)
    
    def byteGetter(blob):
        blob_bytes = blob_service.get_blob_to_bytes(bs_container, blob.name)
        return blob_bytes.content
    choice_blob_bytes = map(byteGetter, choice_blobs)
    
    return zip(choice_blobs, choice_blob_bytes)

def processEntityImages(choice_blobs, db_account_name, db_account_key, ca_file_uri, db_config, cs_account, cs_key, eh_url, eh_account, eh_key):
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
            img_byte = Image.open(io.BytesIO(image_bytes))
            face_iso_image =img_byte.crop((face_rectangle['left'], face_rectangle['top'], face_rectangle['left'] + face_rectangle['width'], 
                face_rectangle['top'] + face_rectangle['height']))

            return face_iso_image
        else:
            return None
    #Convert the base image to PIL object, then detect and extract the face component of the image
    blob_image_faces = list(map(lambda blob_tuple: (blob_tuple[0], Image.open(io.BytesIO(blob_tuple[1])), extractFace(blob_tuple[1])), choice_blobs))
    logging.debug("Face detection run on {0} images".format(len(blob_image_faces)))

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
        subPersonaTableName = os.environ['DB_SUB_PERSONA_TABLE']
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
    rawInsertQuery = session.prepare("INSERT INTO " + rawImageTableName + " (image_id, file_uri, image_bytes) VALUES (?,?,?)")
    refinedInsertQuery = session.prepare("INSERT INTO " + refinedImageTableName + " (image_id, raw_image_edge_id, image_bytes, feature_bytes) VALUES (?,?,?,?)")
    
    client = EventHubClient(eh_url, debug=False, username=eh_account, password=eh_key)
    sender = client.add_sender(partition="0")
    client.run()

    
    face_write_count = 0
    face_label_write_count = 0

    for (blob, image_bytes, face_bytes) in blob_image_faces:
        if face_bytes is not None:
            file_name = blob.name
            (entity, usage, number) = file_name.split('-')
        
            #Generate the face classifier features from the face using VGG-face on Keras
            if face_bytes.mode != "RGB":
                    face_bytes = face_bytes.convert("RGB")
            face_bytes = face_bytes.resize((img_width, img_height))
            image = img_to_array(face_bytes)
            image = np.expand_dims(image, axis=0)
            image = imagenet_utils.preprocess_input(image)
            stuff = vgg_face_feature_gen.predict(image, batch_size=1).flatten()
            writer = avro.io.DatumWriter(schema)
            bytes_writer = io.BytesIO()
            encoder = avro.io.BinaryEncoder(bytes_writer)
            writer.write({"features": stuff.tolist()}, encoder)
            face_feature_bytes = bytes_writer.getvalue()
            #For each image extract the label and use to generate a persona, redundant writes will cancel out

            #Writes the entity label to the persona table 
            #For the time being also write the entity label to the subpersona table and associate with the persona table
            session.execute(personaInsertQuery, (entity,))
            session.execute(subPersonaInsertQuery, (entity, entity))
            logging.info("Writing persona, sub-persona to DB {0}".format(entity))
            
            #Resizes the image to ensure the write query does not exceed maximum size
            width, height = image_bytes.size
            if width > height:
                transform_factor = 256/width
            else:
                transform_factor = 256/height
            compact_image_bytes = image_bytes.resize((round(height*transform_factor), round(width*transform_factor))) 
            
            #Writes the raw image to its table
            imgByteArr = io.BytesIO()
            compact_image_bytes.save(imgByteArr, format='PNG')
            compact_image_bytes = imgByteArr.getvalue()
            hashed_bytes = hashlib.md5(compact_image_bytes).digest()
            hashed_bytes_int = int.from_bytes(hashed_bytes, byteorder='big') #Good identifier, sadly un
            image_hash = str(hashed_bytes_int) #Stupid workaround to Python high precision int incompatability
            session.execute(rawInsertQuery, (image_hash, file_name, compact_image_bytes))
            logging.info("Writing raw image to DB {0}".format(image_hash))
            
            #Resizes the image to ensure the write query does not exceed maximum size
            width, height = face_bytes.size
            if width > height:
                transform_factor = 256/width
            else:
                transform_factor = 256/height
            compact_face_bytes = face_bytes.resize((round(height*transform_factor), round(width*transform_factor))) 
            
            #Write each face extracted from the image to the DB as a refined image
            imgByteArr = io.BytesIO()
            compact_face_bytes.save(imgByteArr, format='PNG')
            compact_face_bytes = imgByteArr.getvalue()
            face_hash_bytes = hashlib.md5(compact_face_bytes).digest()
            face_hashed_bytes_int = int.from_bytes(face_hash_bytes, byteorder='big') #Good identifier, sadly un
            face_hash = str(face_hashed_bytes_int) #Stupid workaround to Python high precision int incompatability
            session.execute(refinedInsertQuery, (face_hash, image_hash, compact_face_bytes, face_feature_bytes))
            logging.info("Writing face image to DB {0}".format(face_hash))
            face_write_count += 1
            
            #If the data is part of the training set, write edges between the sub-personas, face images
            if usage == "Train":
                session.execute(subPersonaEdgeInsertQuery, (entity, face_hash, True))
                sender.send(EventData(json.dumps({"EVENT_TYPE": "LABEL_WRITE", "LABEL_INDEX":face_hash, "WRITE_TIMESTAMP": datetime.datetime.now().timestamp()})))
                logging.info("Writing face label to DB {0}".format(face_hash))
                face_label_write_count += 1
            #Otherwise do not write an edge, these will be predicted later by the training service

    client.stop()
    logging.info("Wrote {0} faces to DB".format(face_write_count))
    logging.info("Wrote {0} face labels to DB".format(face_label_write_count))

def imgPreprocessing(image_file):
    nparr = np.fromstring(image_file, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # cv2.IMREAD_COLOR in OpenCV 3.1
    decoded = cv2.resize(image, (img_width, img_height))
    decoded = decoded.reshape((1,img_width,img_height,3))
    #logging.debug("Converted image from {0} to {1}".format(image.shape, decoded.shape))
    return decoded

def main ():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    FORMAT = '%(asctime) %(message)s'
    logging.basicConfig(format=FORMAT)
    
    #Check if the blob storage access credentials have been loaded as a secret volume use them
    if os.path.exists('/tmp/secrets/bs/blob-storage-account'):
        bs_account = open('/tmp/secrets/bs/blob-storage-account').read()
        bs_key = open('/tmp/secrets/bs/blob-storage-key').read()

        db_account = open('/tmp/secrets/db/db-account').read()
        db_key = open('/tmp/secrets/db/db-key').read()
        
        #Load cognitive service credentials
        cs_account = open('/tmp/secrets/cs/cs-account').read()
        cs_key = open('/tmp/secrets/cs/cs-key').read()
        logging.debug('')

        eh_url = open('').read()
        eh_account = open('').read()
        eh_key = open('').read()
        
    #Otherwise assume it is being run locally and load from environment variables
    else: 
        bs_account = os.environ['BLOB_STORAGE_ACCOUNT']
        bs_key = os.environ['BLOB_STORAGE_KEY']
        
        db_account = os.environ['COSMOS_DB_ACCOUNT']
        db_key = os.environ['COSMOS_DB_KEY']
        
        #Load cognitive service credentials
        cs_account = os.environ['COG_SERV_ACCOUNT']
        cs_key = os.environ['COG_SERV_KEY']
        
        eh_url = os.environ['EVENT_HUB_URL']
        eh_account = os.environ['EVENT_HUB_ACCOUNT']
        eh_key = os.environ['EVENT_HUB_KEY']

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
    processEntityImages(choice_blobs, db_account, db_key, ca_file_uri, db_config, cs_account, cs_key, eh_url, eh_account, eh_key)
    
if __name__ == '__main__':
    '''

    '''
    main()