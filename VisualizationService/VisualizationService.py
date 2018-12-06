from flask import Flask

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from ssl import PROTOCOL_TLSv1_2, CERT_REQUIRED

import json
import logging

import os
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
FORMAT = '%(asctime) %(message)s'
logging.basicConfig(format=FORMAT) 

app = Flask(__name__)
 
#Retrieve db secrets from the kubernates secret volume
db_name = open('/tmp/secrets/db/db-account', 'r').read()
db_passwd = open('/tmp/secrets/db/db-key', 'r').read()

azure_db_domain = ".cassandra.cosmosdb.azure.com"
azure_db_endpoint_uri = db_name + azure_db_domain

#Cassandra connection options for the Azure CosmoDB with Cassandra API from the quickstart documentation on the portal page
#grabbed my CA.pem from Mozilla https://curl.haxx.se/docs/caextract.html
ssl_opts = {
    'ca_certs': './cacert.pem',
    'ssl_version': PROTOCOL_TLSv1_2,
    'cert_reqs': CERT_REQUIRED  # Certificates are required and validated
}
auth_provider = PlainTextAuthProvider(username=db_name, password=db_passwd)
cluster = Cluster([azure_db_endpoint_uri], port = 10350, auth_provider=auth_provider, ssl_options=ssl_opts)
    
cosmos_keyspace = os.environ['DB_KEYSPACE']
    
logging.debug("Attempting to connect to CosmosDB with the following credentials {0}, {1}".format(azure_db_endpoint_uri, db_name, cosmos_keyspace)) 
session = cluster.connect(cosmos_keyspace)

persona_table = os.environ['DB_PERSONA_TABLE']
persona_edge_table = os.environ['DB_PERSONA_EDGE_TABLE']
raw_image_table = os.environ['DB_RAW_IMAGE_TABLE']
refined_image_table = os.environ['DB_REFINED_IMAGE_TABLE']

@app.route('/')
def serveMainPage():
    '''
    
    '''
    logging.debug("mainpage")
    persona_list = list(session.execute("SELECT persona_name FROM " + persona_table))
    page_html = ""
    
    for persona in persona_list:
        page_html += "<div><a href='persona/{0}'>{0}</a></div>\n".format(persona.persona_name)
    
    return page_html
    #return 'give me whales or give me death'

@app.route('/persona/label/<string:persona_name>/<string:img_id>', methods=["POST"])
def relabelImage(persona_name, img_id):
    print("labeling {0}".format(img_id))
    
    session.execute("UPDATE " + persona_edge_table + " SET label_assoc_flag=%s WHERE persona_name=%s AND assoc_image_id=%s", (True, persona_name, img_id))
    
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
    image_list = list(session.execute("SELECT assoc_image_id, label_assoc_flag FROM " + persona_edge_table + " WHERE persona_name=%s", (persona_name,)))
    page_html = ""
    for image in image_list:
        page_html += "<div><img src='img/{0}':image/jpeg;base64;+encoded_string></img>{1}<form action='label/{2}/{0}' method='post'><button type='submit'>retrain</button></form></div>".format(image.assoc_image_id, "labeled" if image.label_assoc_flag else "predicted", persona_name)
    
    return page_html

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')