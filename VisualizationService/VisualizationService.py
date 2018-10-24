from flask import Flask

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from ssl import PROTOCOL_TLSv1_2, CERT_REQUIRED

import json

app = Flask(__name__)
 
cfg = json.load(open("VisualizationServiceConfig.json", "r"))
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

personaTableName = cosmoCfg['personaTableName']
personaEdgeTableName = cosmoCfg['personaEdgeTableName']
rawImageTableName = cosmoCfg['rawImageTableName']
refinedImageTableName = cosmoCfg['refinedImageTableName']

@app.route('/')
def serveMainPage():
    print("mainpage")
    persona_list = list(session.execute("SELECT persona_name FROM " + personaTableName))
    page_html = ""
    
    for persona in persona_list:
        page_html += "<div><a href='persona/{0}'>{0}</a></div>\n".format(persona.persona_name)
    
    return page_html
    #return 'give me whales or give me death'
    
@app.route('/persona/label/<string:persona_name>/<string:img_id>', methods=["POST"])
def relabelImage(persona_name, img_id):
    print("labeling {0}".format(img_id))
    
    session.execute("UPDATE " + personaEdgeTableName + " SET label_assoc_flag=%s WHERE persona_name=%s AND assoc_image_id=%s", (True, persona_name, img_id))
    
    return "Retrained!"

@app.route('/persona/img/<string:img_id>')
def servePersonaRawImg(img_id):
    print(img_id)
    
    #Approach to hosting images in DB https://stackoverflow.com/questions/49795388/how-to-show-image-from-mysql-database-in-flask
    
    all_raw_images = list(session.execute("SELECT image_bytes FROM " + rawImageTableName + " WHERE image_id=%s", (img_id,)))
    first_image = all_raw_images[0].image_bytes
    
    return first_image

#Profile page for an individual persona selected from the homepage
@app.route('/persona/<string:persona_name>')
def servePersonaPage(persona_name):
    image_list = list(session.execute("SELECT assoc_image_id, label_assoc_flag FROM " + personaEdgeTableName + " WHERE persona_name=%s", (persona_name,)))
    page_html = ""
    for image in image_list:
        page_html += "<div><img src='img/{0}':image/jpeg;base64;+encoded_string></img>{1}<form action='label/{2}/{0}' method='post'><button type='submit'>retrain</button></form></div>".format(image.assoc_image_id, "labeled" if image.label_assoc_flag else "predicted", persona_name)
    
    return page_html

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
