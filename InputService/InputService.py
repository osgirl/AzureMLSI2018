'''
    This script runs once upon initial instantiation of the container to load the test data.
    The container can be run without running this script. 
'''
from azure.storage.file import FileService
from azure.storage.file import ContentSettings
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from ssl import PROTOCOL_TLSv1_2, CERT_REQUIRED

import random
import os

import hashlib

import json

def testPopulateCassandraTables(session, cosmoCfg):
    '''
    Populates configured CosmoDB Cassandra Tables with test data taken from the configured Azure
    storage account
    '''
    personaTableName = cosmoCfg['personaTableName']
    personaEdgeTableName = cosmoCfg['personaEdgeTableName']
    rawImageTableName = cosmoCfg['rawImageTableName']
    refinedImageTableName = cosmoCfg['refinedImageTableName']

    #Iterates through labeled presidential images in a provided directory, extracts the label from 
    #the filename, and adds them as (randomly) labeled or unlabeled to the database
    personaInsertQuery = session.prepare("INSERT INTO " + personaTableName + " (persona_name) VALUES (?)")
    personaEdgeInsertQuery = session.prepare("INSERT INTO " + personaEdgeTableName + " (persona_name, assoc_image_id, label_assoc_flag, pred_assoc_flag) VALUES (?,?,?,?)")
    rawInsertQuery = session.prepare("INSERT INTO " + rawImageTableName + " (image_id, refined_image_edge_id, file_uri, image_bytes) VALUES (?,?,?,?)")
    refinedInsertQuery = session.prepare("INSERT INTO " + refinedImageTableName + " (image_id, raw_image_edge_id, image_bytes) VALUES (?,?,?)")

    for dirPath, dirNames, fileNames in os.walk("./TestImages"):
        for fileName in fileNames:
            print(fileName)
            fileHandle = open(dirPath + "/" + fileName, 'rb') #TODO make more robust with a non-OS specific seperator
            label = fileName.split("-")[0]
            choice = random.choice((0,1,2))
            
            #For each image extract the label and use to generate a persona, redundant writes will cancel out
            image_bytes = fileHandle.read();
            hashed_bytes = hashlib.md5(image_bytes).digest()
            hashed_bytes_int = int.from_bytes(hashed_bytes, byteorder='big') #Good identifier, sadly un
            hashed_bytes_str = str(hashed_bytes_int) #Stupid workaround to Python high precision int incompatability
            
            #
            session.execute(personaInsertQuery, (label,))
            session.execute(rawInsertQuery, (hashed_bytes_str, "", fileName, image_bytes))
            session.execute(refinedInsertQuery, (hashed_bytes_str, "", image_bytes))
            
            #Add image rows
            session.execute(refinedInsertQuery, (hashed_bytes_str, hashed_bytes_str, b""))
            
            #For one out of every three images register as label associated, otherwise predictive
            if choice == 2:
                session.execute(personaEdgeInsertQuery, (label, hashed_bytes_str, True, False))
            else:
                session.execute(personaEdgeInsertQuery, (label, hashed_bytes_str, False, True))
            
            #fileService.create_file_from_path(cosmoCfg.fileStorageShareName, cosmoCfg.fileStorageDir, fileName, dirPath + "/" + fileName)
            fileHandle.close()

if __name__ == '__main__':
    '''
    fileService = FileService(account_name=cosmoCfg[fileAccountName], account_key=cosmoCfg[fileAccountSecret])

    #Get 
    for file in fileService.list_directories_and_files(cosmoCfg[shareName], cosmoCfg[dirName]):
        print(file.name)

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