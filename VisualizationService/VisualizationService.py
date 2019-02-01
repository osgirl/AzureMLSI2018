from flask import Flask, render_template, jsonify, url_for

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from ssl import PROTOCOL_TLSv1_2, CERT_REQUIRED

import json
import yaml
import logging

import hashlib

import os
import sys

from PIL import Image

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
FORMAT = '%(asctime) %(message)s'
logging.basicConfig(format=FORMAT) 

app = Flask(__name__)
 
#Retrieve db secrets from the kubernates secret volume
if os.path.exists('/tmp/secrets/db/db-account'):
    db_account_name = open('/tmp/secrets/db/db-account', 'r').read()
    db_account_key = open('/tmp/secrets/db/db-key', 'r').read()
    logging.debug('Loaded db secrets from secret volume')

    cosmos_keyspace = os.environ['DB_KEYSPACE']
    persona_table = os.environ['DB_PERSONA_TABLE']
    sub_persona_table = os.environ['DB_SUB_PERSONA_TABLE']
    sub_persona_face_edge_table = os.environ['DB_SUB_PERSONA_FACE_EDGE_TABLE']
    face_sub_persona_edge_table = os.environ['DB_FACE_SUB_PERSONA_EDGE_TABLE']
    raw_image_table = os.environ['DB_RAW_IMAGE_TABLE']
    face_image_table = os.environ['DB_FACE_IMAGE_TABLE']
#Otherwise run locally and retrieve from project cluster definition file
else: 
    db_account_name = os.environ['COSMOS_DB_ACCOUNT']
    db_account_key = os.environ['COSMOS_DB_KEY']
    logging.debug('Loaded db secrets from test environment variables')
    
    merged_config = yaml.safe_load_all(open("../cluster-deployment.yml"))
    for config in merged_config:
        if config['metadata']['name'] == 'azmlsi-db-config':
            db_config  = config['data']
            
    cosmos_keyspace = db_config['db-keyspace']
    persona_table = db_config['db-persona-table']
    sub_persona_table = db_config['db-sub-persona-table']
    sub_persona_face_edge_table = db_config['db-sub-persona-face-edge-table']
    face_sub_persona_edge_table = db_config['db-face-sub-persona-edge-table']
    raw_image_table = db_config['db-raw-image-table']
    face_image_table = db_config['db-face-image-table']
    
#Build connection string
azure_db_domain = ".cassandra.cosmosdb.azure.com"
azure_db_endpoint_uri = db_account_name + azure_db_domain

#Cassandra connection options for the Azure CosmoDB with Cassandra API from the quickstart documentation on the portal page
#grabbed my CA.pem from Mozilla https://curl.haxx.se/docs/caextract.html
ssl_opts = {
    'ca_certs': './cacert.pem',
    'ssl_version': PROTOCOL_TLSv1_2,
    'cert_reqs': CERT_REQUIRED  # Certificates are required and validated
}
auth_provider = PlainTextAuthProvider(username=db_account_name, password=db_account_key)
cluster = Cluster([azure_db_endpoint_uri], port = 10350, auth_provider=auth_provider, ssl_options=ssl_opts)
    
logging.info("Attempting to connect to CosmosDB with the following credentials {0}, {1}".format(azure_db_endpoint_uri, db_account_name, cosmos_keyspace)) 
session = cluster.connect(cosmos_keyspace)

def imageHashAndCache(image_bytes):
    '''
        Methods to 
    '''
    image_hash = hashlib.md5(image_bytes).hexdigest()  
    image_hash_path = os.path.join("./static/img/", image_hash + ".jpg")
    image_url = url_for('static', filename="img/" + image_hash + ".jpg")
    logging.debug("image static url is {0}".format(image_url))
    
    #Check to see if the image is locally cached, add to cache if not
    if os.path.exists(image_hash_path):
        logging.debug("{0} exists".format(image_hash_path))
    else:
        file_handle = open(image_hash_path, 'wb')
        file_handle.write(image_bytes)
        file_handle.close()
    
    return image_url

@app.route('/')
def serveMainPage():
    '''
        GET for the main page which lists available entities, thumbnails and their labeled, predicted image_bytes counts
    '''
    
    #Grab list of current entities
    sub_persona_query = session.prepare("SELECT sub_persona_name FROM " + sub_persona_table + " WHERE persona_name=?")

    persona_list = list(session.execute("SELECT persona_name FROM " + persona_table))
    sub_persona_list = []
    for persona in persona_list:
        sub_persona_list += list(session.execute(sub_persona_query, (persona.persona_name,)))
    logging.info("Loading {0} sub-personas from DB table {1}".format(len(sub_persona_list), persona_table))
    
    thumbnail_path_list = []
    number_of_predicted_list = []
    number_of_labeled_list = []
    for persona in persona_list:
        #Pivot through entity-image table to recover entity image byte blobs
        image_id_list = session.execute("SELECT sub_persona_name, assoc_face_id, label_v_predict_assoc_flag FROM {0} WHERE sub_persona_name='{1}'".format(sub_persona_face_edge_table, persona.persona_name))
        labeled_faces = list(filter(lambda x: x.label_v_predict_assoc_flag == True, image_id_list))
        predicted_faces = list(filter(lambda x: x.label_v_predict_assoc_flag == False, image_id_list))
        number_of_labeled_images = len(labeled_faces)
        number_of_predicted_images = len(predicted_faces)
        logging.debug(number_of_labeled_images)
        logging.debug(labeled_faces[0].assoc_face_id)

        image_bytes = session.execute("SELECT face_id, face_bytes FROM {0} WHERE face_id='{1}'".format(face_image_table, labeled_faces[0].assoc_face_id))
        image_bytes = list(map(lambda x: x.face_bytes, image_bytes))
        
        image_url = imageHashAndCache(image_bytes[0])
            
        thumbnail_path_list.append(image_url)
        number_of_labeled_list.append(number_of_labeled_images)
        number_of_predicted_list.append(number_of_predicted_images)
    
    name_list = map(lambda x: x.persona_name, persona_list)
    persona_profiles_list = zip(name_list, thumbnail_path_list, number_of_labeled_list, number_of_predicted_list)
    
    return render_template('index.html', persona_profiles_list=persona_profiles_list)

@app.route('/entities/<string:persona_name>')
def serveEntityPage(persona_name):
    '''
        Get for the entity page which shows the entity, sub-entities and labeled/predicted images associated with
        each sub-entity; label images associated with that entity
    '''
    image_id_list = list(session.execute("SELECT sub_persona_name, assoc_face_id, label_v_predict_assoc_flag FROM {0} WHERE sub_persona_name='{1}'".format(sub_persona_face_edge_table, persona_name)))
    labeled_faces = list(filter(lambda x: x.label_v_predict_assoc_flag == True, image_id_list))
    predicted_faces = list(filter(lambda x: x.label_v_predict_assoc_flag == False, image_id_list))
    logging.info("Retrieved {0} labeled images, {1} predicted images, {2} total for persona {3}".format(len(labeled_faces), len(predicted_faces), len(image_id_list), persona_name))

    def retrieveAndCacheFaces(face_id_row):
        logging.info("Retrieving bytes for face id {0} from DB".format(face_id_row.assoc_face_id))
        image_byte = session.execute("SELECT face_id, face_bytes FROM {0} WHERE face_id='{1}'".format(face_image_table, face_id_row.assoc_face_id))
        image_byte = list(map(lambda x: x.face_bytes, image_byte))
        if len(image_byte) > 0:
            image_url = imageHashAndCache(image_byte[0])
            return {"face_id": face_id_row.assoc_face_id, "face_url": image_url}
        else:
            return None
    
    predicted_image_urls = map(retrieveAndCacheFaces, predicted_faces) 
    labeled_image_urls = map(retrieveAndCacheFaces, labeled_faces)
    return render_template('entity.html', labeled_image_urls=labeled_image_urls, predicted_image_urls=predicted_image_urls)

@app.route('/images/<string:image_id>')
def serveImagePage(image_id):
    '''
        Get for an individual image showing its original image, extracted face, associated entity/sub-entity and 
        explanatory image
    '''
    
    results = session.execute("SELECT sub_persona_name, assoc_face_id,  label_v_predict_assoc_flag FROM " + face_sub_persona_edge_table + " WHERE assoc_face_id='{0}'".format(image_id))
    
        
    if len(list(filter(lambda x: x.label_v_predict_assoc_flag == True, results))) == 1:
        logging.info("Labeled Edge Found")
        
    else:
        logging.info("No Labeled Edge")
    
    
    return render_template("image.html")

@app.route('/addImage')
def serveAppendImagePage():
    '''
        Get for the page providing the forms to submit a new image either labeled with an existing entity, new entity
        and sub-entity thereof
    '''
    return None
'''
@app.route('/persona/label/<string:persona_name>/<string:img_id>', methods=["POST"])
def relabelImage(persona_name, img_id):
    print("labeling {0}".format(img_id))
    
    session.execute("UPDATE " + sub_persona_edge_table + " SET label_v_predict_assoc_flag=%s WHERE persona_name=%s AND assoc_face_id=%s", (True, persona_name, img_id))
    
    return "Retrained!"

@app.route('/persona/img/<string:img_id>')
def servePersonaRawImg(img_id):
    print(img_id)
    
    #Approach to hosting images in DB https://stackoverflow.com/questions/49795388/how-to-show-image-from-mysql-database-in-flask
    
    all_raw_images = list(session.execute("SELECT image_bytes FROM " + raw_image_table + " WHERE image_id=%s", (img_id,)))
    first_image = all_raw_images[0].image_bytes
    
    return first_image

#Profile page for an individual persona selected from the homepage
@app.route('/persona/<string:persona_name>')
def servePersonaPage(persona_name):
    image_list = list(session.execute("SELECT assoc_face_id, label_v_predict_assoc_flag FROM " + sub_persona_edge_table + " WHERE persona_name=%s", (persona_name,)))
    page_html = ""
    for image in image_list:
        page_html += "<div><img src='img/{0}':image/jpeg;base64;+encoded_string></img>{1}<form action='label/{2}/{0}' method='post'><button type='submit'>retrain</button></form></div>".format(image.assoc_face_id, "labeled" if image.label_v_predict_assoc_flag else "predicted", persona_name)
    
    return page_html
'''

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')