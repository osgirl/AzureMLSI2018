# TensorFlow and tf.keras
''''
Draws from the example https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/5.3-using-a-pretrained-convnet.ipynb


'''
import tensorflow as tf
import keras

import os
import sys
import yaml
import logging
import time

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from ssl import PROTOCOL_TLSv1_2, CERT_REQUIRED

import onnxmltools

# Helper libraries
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
#import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

from keras import backend as K
from keras.applications import VGG16

import avro.schema, avro.io
import io
from azure.eventhub import EventHubClient, EventData, Receiver, Offset

from azureml.core.workspace import Workspace

import json
import datetime

def isNewLabeledData(eh_url, eh_offset_url, eh_account, eh_key):
    '''
    Examines the EventHub to identify if sufficient quantities of new training data is available to trigger a re-train
    ''' 
    
    CONSUMER_GROUP = "$default"
    PARTITION = "0"
    
    offset_client = EventHubClient(eh_offset_url, debug=False, username=eh_account, password=eh_key)
    offset_receiver = offset_client.add_receiver(CONSUMER_GROUP, PARTITION, prefetch=5000)
    offset_sender = offset_client.add_sender(partition="0")
    offset_client.run()

    #Retrieves the current offset/sequence number for the write event queue from the dedicated offset queue
    offsets = offset_receiver.receive(timeout=50)
    current_offset = -1 #Default to -1 or reading the entire feed if another offset is not retrieved
    logging.info(len(offsets))
    for offset in offsets:
        offset_event = json.loads(offset.body_as_str())
        current_offset = offset_event['CURRENT_OFFSET']
        logging.info("Retrieved previous offset event {0}".format(offset_event))
    
    current_offset = -1
    
    #Use the retrieved offset/sequence number to retrieve new writes
    event_client = EventHubClient(eh_url, debug=False, username=eh_account, password=eh_key)
    receiver = event_client.add_receiver(CONSUMER_GROUP, PARTITION, prefetch=5000, offset=Offset(current_offset))
    event_client.run()
    batch = receiver.receive(timeout=5000)
    new_label_count = len(batch)
    for stuff in batch:
        logging.info("Offset {0}".format(stuff.sequence_number))
        current_offset = int(stuff.sequence_number) if int(stuff.sequence_number) > current_offset else current_offset
        logging.info("Message {0}".format(stuff.body_as_str()))
    logging.info("Processed {0} new label writes".format(new_label_count))
    
    #Write the last retrieved offset/sequence number to the offset message queue to be used in the next read
    offset_sender.send(EventData(json.dumps({"TIMESTAMP": datetime.datetime.now().timestamp(), "CURRENT_OFFSET": current_offset})))
    logging.info("Stored current offset event {0}".format(current_offset))
    #sender.send(EventData(json.dumps({"EVENT_TYPE": "LABEL_WRITE", "LABEL_INDEX":face_hash, "WRITE_TIMESTAMP": datetime.datetime.now().timestamp()})))
    
    #Close queue clients
    offset_client.stop()
    event_client.stop()
    
    #Return true if more then results found to execute retrain
    return True if new_label_count > 5 else False

def retrainClassifier(session, db_config):
    '''
    Gathers available labeled faces from the database for personas, sub-personas to train a new keras classifier
    '''
    (keyspace, personaTableName, subPersonaTableName, subPersonaFaceEdgeTableName, faceSubPersonaEdgeTableName, rawImageTableName, faceImageTableName) = getTables(db_config)

    #Prepare Cosmos DB session and insertion queries
    persona_list = list(session.execute("SELECT persona_name FROM " + personaTableName))
    unique_features_set = set()
    features_list = []
    labels_list = []
    for persona in persona_list:
        logging.info("Retrieving features for {0}".format(persona.persona_name))
        image_id_list = session.execute("SELECT sub_persona_name, assoc_face_id, label_v_predict_assoc_flag FROM {0} WHERE sub_persona_name='{1}'".format(subPersonaFaceEdgeTableName, persona.persona_name))

        for image_id in image_id_list:
            image_features = session.execute("SELECT face_id, face_bytes, feature_bytes FROM {0} WHERE face_id='{1}'".format(faceImageTableName, image_id.assoc_face_id))
            for image_byte in image_features:
                schema = avro.schema.Parse(open("./VGGFaceFeatures.avsc", "rb").read())
                bytes_reader = io.BytesIO(image_byte.feature_bytes)
                decoder = avro.io.BinaryDecoder(bytes_reader)
                reader = avro.io.DatumReader(schema)
                features = reader.read(decoder)
                
                unique_features_set.add(persona.persona_name)
                features_list.append(features['features'])
                labels_list.append(persona.persona_name)
    
        #Convert the features and labels into keras compatible structures, train dense NN classifier
        unique_features_list = list(unique_features_set)
        homogenized_label_list = list(map(lambda x: unique_features_list.index(x), labels_list))
        model = Sequential()
        model.add(Dense(1024, input_dim=512, activation='relu')) #we add dense layers so that the model can learn more complex functions and classify for better results.
        model.add(Dense(1024,activation='relu')) #dense layer 2
        model.add(Dense(512,activation='relu')) #dense layer 3
        model.add(Dense(1,activation='sigmoid')) #final layer with softmax activation
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(x=np.array(features_list), y=np.array(homogenized_label_list), epochs=5, batch_size=1)
        scores = model.evaluate(np.array(features_list), np.array(homogenized_label_list))

        onnx_model = onnxmltools.convert_keras(model)
        onnxmltools.utils.save_model(onnx_model, 'example.onnx')
        
        return model
    
def getTables(db_config=None):
    '''
    
    '''
    if db_config is None:
        keyspace = os.environ['DB_KEYSPACE']
        personaTableName = os.environ['DB_PERSONA_TABLE']
        subPersonaTableName = os.environ['DB_SUB_PERSONA_TABLE']
        subpersonaEdgeTableName = os.environ['DB_SUB_PERSONA_EDGE_TABLE']
        rawImageTableName = os.environ['DB_RAW_IMAGE_TABLE']
        faceImageTableName = os.environ['DB_REFINED_IMAGE_TABLE']
    #Otherwise load db config
    else:
        keyspace = db_config['db-keyspace']
        personaTableName = db_config['db-persona-table']
        subPersonaTableName = db_config['db-sub-persona-table']
        subPersonaFaceEdgeTableName = db_config['db-sub-persona-face-edge-table']
        faceSubPersonaEdgeTableName = db_config['db-face-sub-persona-edge-table']
        rawImageTableName = db_config['db-raw-image-table']
        faceImageTableName = db_config['db-face-image-table']
    return (keyspace, personaTableName, subPersonaTableName, subPersonaFaceEdgeTableName, faceSubPersonaEdgeTableName, rawImageTableName, faceImageTableName)

def repredictImages(session, model, db_config):
    #If no db config file is passed, look for the container environment variables
    (keyspace, personaTableName, subPersonaTableName, subPersonaFaceEdgeTableName, faceSubPersonaEdgeTableName, rawImageTableName, faceImageTableName) = getTables(db_config)
    
    #Collect a list of the labeled facial images
    labeled_face_ids = session.execute("SELECT sub_persona_name, assoc_face_id, label_v_predict_assoc_flag FROM {0}".format(subPersonaFaceEdgeTableName))
    labeled_face_ids = filter(lambda x: x.label_v_predict_assoc_flag == True, labeled_face_ids)
    labeled_face_ids = list(map(lambda x: x.assoc_face_id, labeled_face_ids))
    logging.info("Retrieved {0} labeled image ids".format(len(labeled_face_ids)))
    
    #Collect facial images which are not labeled to be re-predicted
    #list(session.execute("SELECT assoc_face_id FROM {0} WHERE sub_persona_name={1}".format(faceSubPersonaEdgeTableName, "TEMPORARY")))
    unlabeled_face_ids = list(session.execute("SELECT face_id FROM {0}".format(faceImageTableName)))
    logging.info("Retrieved {0} unlabeled face ids".format(len(unlabeled_face_ids)))
    unlabeled_face_ids = list(filter(lambda x: x.face_id not in labeled_face_ids, unlabeled_face_ids))
    logging.info("Retrieved {0} unlabeled face ids".format(len(unlabeled_face_ids)))
    
    '''
    unlabeled_images = list(filter(lambda x: x.image_id not in labeled_face_ids, unlabeled_images))
    logging.info("Retrieved {0} unlabeled images".format(len(unlabeled_images)))
    
    #Label unlabeled images
    def predictImage(image_features):
        schema = avro.schema.Parse(open("./VGGFaceFeatures.avsc", "rb").read())
        bytes_reader = io.BytesIO(image_features.feature_bytes)
        decoder = avro.io.BinaryDecoder(bytes_reader)
        reader = avro.io.DatumReader(schema)
        features = reader.read(decoder)
        
        #
        prediction = model.predict(features)
        logging.info(prediction)
        return prediction
    prediction_labels = map(predictImage, unlabeled_images)
    '''
    return True
        
def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    FORMAT = '%(asctime) %(message)s'
    logging.basicConfig(format=FORMAT)
    
    ca_file_uri = "./cacert.pem"

    #Check if the blob storage access credentials have been loaded as a secret volume use them
    if os.path.exists('/tmp/secrets/db/db-account'):
        db_account_name = open('/tmp/secrets/db/db-account').read()
        db_account_key = open('/tmp/secrets/db/db-key').read()
        
        eh_url = open().read()
        eh_offset_url = open().read()
        eh_account = open().read()
        eh = open().read()
        
    #Otherwise assume it is being run locally and load from environment variables
    else: 
        db_account = os.environ['COSMOS_DB_ACCOUNT']
        db_key = os.environ['COSMOS_DB_KEY']

        eh_url = os.environ['EVENT_HUB_URL']
        eh_offset_url = os.environ['EVENT_HUB_OFFSET_URL']
        eh_account = os.environ['EVENT_HUB_ACCOUNT']
        eh_key = os.environ['EVENT_HUB_KEY']
        
    #If the test container is not loaded as an environment variable, assume a local run
    #and use the configuration information in the deployment config
    if 'DB_PERSONA_EDGE_TABLE' in os.environ:
        db_config = None
    else:
        merged_config = yaml.safe_load_all(open("../cluster-deployment.yml"))
        for config in merged_config:
            if config['metadata']['name'] == 'azmlsi-db-config':
                db_config  = config['data']
    
    #If no db config file is passed, look for the container environment variables
    if db_config is None:
        keyspace = os.environ['DB_KEYSPACE']
    #Otherwise load db config
    else:
        keyspace = db_config['db-keyspace']
    
    #Connect to the database
    ssl_opts = {
        'ca_certs': ca_file_uri,
        'ssl_version': PROTOCOL_TLSv1_2,
        'cert_reqs': CERT_REQUIRED  # Certificates are required and validated
    }
    auth_provider = PlainTextAuthProvider(username=db_account, password=db_key)
    endpoint_uri = db_account + '.cassandra.cosmosdb.azure.com'
    cluster = Cluster([endpoint_uri], port = 10350, auth_provider=auth_provider, ssl_options=ssl_opts)
    session = cluster.connect(keyspace);   
     
    #Workspace.get("azmlsiws", subscription_id="0b753943-9062-4ec0-9739-db8fb455aeba", resource_group="CommercialCyberAzure")
                
    if isNewLabeledData(eh_url, eh_offset_url, eh_account, eh_key):
        model = retrainClassifier(session, db_config)
        repredictImages(session, model, db_config)
if __name__ == '__main__':
    '''

    '''
    main()