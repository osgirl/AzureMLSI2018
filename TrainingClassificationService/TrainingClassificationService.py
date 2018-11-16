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

# Helper libraries
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
#import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

from keras import backend as K
from keras.applications import VGG16



def extract_features(input_images, sample_count, datagen, batch_size):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    '''
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')
    '''
    
    decoded_input = []
    for image in input_images:
        decoded = cv2.imdecode(np.frombuffer(image, np.uint8), -1)
        decode_input.append(decoded)
    
    generator = datagen.flow(
        decoded_input,
        #target_size=(150, 150),
        batch_size=batch_size)
        #class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            # Note that since generators yield data indefinitely in a loop,
            # we must `break` after every image has been seen once.
            break
    return features, labels

def getTrainingFaces(db_account, db_key, ca_file_uri, db_config):

    ssl_opts = {
        'ca_certs': ca_file_uri,
        'ssl_version': PROTOCOL_TLSv1_2,
        'cert_reqs': CERT_REQUIRED  # Certificates are required and validated
    }
    auth_provider = PlainTextAuthProvider(username=db_account, password=db_key)
    endpoint_uri = db_account + '.cassandra.cosmosdb.azure.us'
    logging.debug("Connecting to CosmosDB Cassandra using {0} {1} {2} {3}".format(db_account, db_key, endpoint_uri, os.path.exists(ca_file_uri)))

    cluster = Cluster([endpoint_uri], port = 10350, auth_provider=auth_provider, ssl_options=ssl_opts)

    if db_config is None:
        keyspace = os.environ['DB_KEYSPACE']
        persona_table_name = os.environ['DB_PERSONA_TABLE']
        persona_edge_table_name = os.environ['DB_PERSONA_EDGE_TABLE']
        rawImageTableName = os.environ['DB_RAW_IMAGE_TABLE']
        refinedImageTableName = os.environ['DB_REFINED_IMAGE_TABLE']
    #Otherwise load db config
    else:
        keyspace = db_config['db-keyspace']
        persona_table_name = db_config['db-persona-table']
        persona_edge_table_name = db_config['db-persona-edge-table']
        rawImageTableName = db_config['db-raw-image-table']
        refinedImageTableName = db_config['db-refined-image-table']

    session = cluster.connect(keyspace);   

    personas = list(session.execute("SELECT persona_name FROM " + persona_table_name))
    training_dictionary = {}
    for persona in personas:
        persona_name = persona.persona_name
        print(persona_name)
        results = list(session.execute("SELECT persona_name, assoc_face_id FROM " + persona_edge_table_name + " WHERE persona_name=%s", (persona_name,)))
        face_ids = list(map(lambda x: x.assoc_face_id, results))
        face_bytes = []
        for face_id in face_ids:
            results = list(session.execute("SELECT image_bytes FROM " + refinedImageTableName + " WHERE image_id=%s", (face_id,)))
            face_bytes.append(results[0].image_bytes)
            
        logging.debug("Retrieved {0} for persona {1}".format(persona_name, str(len(face_bytes))))
        training_dictionary[persona_name] = face_bytes
    return training_dictionary

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
    
    getTrainingFaces(db_account_name, db_account_key, ca_file_uri, db_config)
    
    #K.set_image_dim_ordering('th')

    conv_base = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(150, 150, 3))

    print(conv_base.summary())

    #Originally configured to draw from local files for the training and validation set
    #train_dir = './TestImages3/Train'
    #validation_dir = './TestImages3/Validate'
    train_dir = './TestImages/Train'
    validation_dir = './TestImages/Validate'
     
    nTrain = 600
    nVal = 150

    datagen = ImageDataGenerator(rescale=1./255)
    batch_size = 20
    train_features, train_labels = extract_features(train_dir, 10, datagen, batch_size)
    '''
    validation_features, validation_labels = extract_features(validation_dir, 19)
    #test_features, test_labels = extract_features(test_dir, 1000)

    train_features = np.reshape(train_features, (20, 4 * 4 * 512))
    validation_features = np.reshape(validation_features, (19, 4 * 4 * 512))

    from keras import models
    from keras import layers
    from keras import optimizers

    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
                  loss='binary_crossentropy',
                  metrics=['acc'])

    history = model.fit(train_features, train_labels,
                        epochs=30,
                        batch_size=20,
                        validation_data=(validation_features, validation_labels))

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")
    '''
    
if __name__ == '__main__':
    '''

    '''
    main()