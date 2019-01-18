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

def main():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    FORMAT = '%(asctime) %(message)s'
    logging.basicConfig(format=FORMAT)
    
    ca_file_uri = "./cacert.pem"

    #Check if the blob storage access credentials have been loaded as a secret volume use them
    if os.path.exists('/tmp/secrets/db/db-account'):
        db_account_name = open('/tmp/secrets/db/db-account').read()
        db_account_key = open('/tmp/secrets/db/db-key').read()
        logging.debug('Loaded db secrets from secret volume')
    #Otherwise assume it is being run locally and load from environment variables
    else: 
        db_account_name = os.environ['AZ_DB_ACCOUNT']
        db_account_key = os.environ['AZ_DB_KEY']

    #If the test container is not loaded as an environment variable, assume a local run
    #and use the configuration information in the deployment config
    if 'DB_PERSONA_EDGE_TABLE' in os.environ:
        db_config = None
    else:
        merged_config = yaml.safe_load_all(open("../cluster-deployment.yml"))
        for config in merged_config:
            if config['metadata']['name'] == 'azmlsi-db-config':
                db_config  = config['data']
    
    
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
    
    persona_list = list(session.execute("SELECT persona_name FROM " + personaTableName))
    
    unique_features_set = set()
    features_list = []
    labels_list = []
    for persona in persona_list:
        logging.debug("Retrieving features for {0}".format(persona.persona_name))
        image_id_list = session.execute("SELECT sub_persona_name, assoc_face_id, label_v_predict_assoc_flag FROM {0} WHERE sub_persona_name='{1}'".format(subPersonaEdgeTableName, persona.persona_name))

        for image_id in image_id_list:
            image_bytes = session.execute("SELECT image_id, image_bytes, feature_bytes FROM {0} WHERE image_id='{1}'".format(refinedImageTableName, image_id.assoc_face_id))
            logging.debug("stuff")
            for image_byte in image_bytes:
                schema = avro.schema.Parse(open("./VGGFaceFeatures.avsc", "rb").read())
                bytes_reader = io.BytesIO(image_byte.feature_bytes)
                decoder = avro.io.BinaryDecoder(bytes_reader)
                reader = avro.io.DatumReader(schema)
                features = reader.read(decoder)
                
                unique_features_set.add(persona.persona_name)
                features_list.append(features['features'])
                labels_list.append(persona.persona_name)
    
        print(features_list)
        
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
    
if __name__ == '__main__':
    '''

    '''
    main()