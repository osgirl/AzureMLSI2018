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

import InitInputServiceConfig as cfg

if __name__ == '__main__':
    '''
    fileService = FileService(account_name=cfg[fileAccountName], account_key=cfg[fileAccountSecret])

    #Get 
    for file in fileService.list_directories_and_files(cfg[shareName], cfg[dirName]):
        print(file.name)

    '''
    
    
    if cfg.localDBEndpoint:
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
        auth_provider = PlainTextAuthProvider(username=cfg.cosmoDBUsername, password=cfg.cosmoDBSecret)
        cluster = Cluster([cfg.cosmoDBEndpointUri], port = 10350, auth_provider=auth_provider, ssl_options=ssl_opts)
        
        
        
    #Checks to ensure that the demonstration keyspace exists and creates it if not 
    session = cluster.connect()
    print("\nCreating Keyspace")
    if cfg.localDBEndpoint:
        #Modified local keyspace settings
        session.execute('CREATE KEYSPACE IF NOT EXISTS ' + cfg.cosmoDBKeyspaceName + ' WITH replication = {\'class\' : \'SimpleStrategy\', \'replication_factor\' : 1 }')
        session.shutdown()
    else:
        #Default keyspace settings from Azure CosmoDB
        session.execute('CREATE KEYSPACE IF NOT EXISTS ' + cfg.cosmoDBKeyspaceName + ' WITH replication = {\'class\': \'NetworkTopologyStrategy\', \'datacenter\' : 1 }'); #Formatting is stupid on this string due to the additional curley braces
    
    session = cluster.connect(cfg.cosmoDBKeyspaceName)
    
    personaTableName = 'personaTable'
    personaEdgeTableName = 'personaEdgeTable'
    rawImageTableName = 'rawImageTable'
    refinedImageTableName = 'refinedImageTable'
    
    def demoTableGenerator():
        #Create table
        print("\nCreating Table")
        session.execute('DROP TABLE IF EXISTS ' + personaTableName)
        session.execute('DROP TABLE IF EXISTS ' + personaEdgeTableName)
        session.execute('DROP TABLE IF EXISTS ' + rawImageTableName)
        session.execute('DROP TABLE IF EXISTS ' + refinedImageTableName)

        #Persona table provides metadata on each persona in the demonstration database such as name and date of birth
        #This table is set up primarily to be scanned and used to pivot to the persona edge table to discover images 
        session.execute('CREATE TABLE IF NOT EXISTS ' + personaTableName + ' (persona_name text, PRIMARY KEY(persona_name))');
        
        #Persona edge table contains the associations to pivot from a selected persona to its associated images
        #These associations can exist either due to explicit labeling or predicted labels
        session.execute('CREATE TABLE IF NOT EXISTS ' + personaEdgeTableName + ' (persona_name text, assoc_image_id text, label_assoc_flag boolean, pred_assoc_flag boolean, PRIMARY KEY(persona_name, assoc_image_id))');
        
        #Refined table stores extracted face blobs and associative edges to the raw image from which it was derived
        session.execute('CREATE TABLE IF NOT EXISTS ' + refinedImageTableName + ' (image_id text, raw_image_edge_id text, image_bytes blob, PRIMARY KEY(image_id))');
        
        #Raw table stores pre-extraction images that contain at least one face
        session.execute('CREATE TABLE IF NOT EXISTS ' + rawImageTableName + ' (image_id text, refined_image_edge_id text, file_uri text, image_bytes blob, PRIMARY KEY(image_id))');

    #Setup Tables
    demoTableGenerator()
    

    def testGenerateTables(session):
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
                
                #fileService.create_file_from_path(cfg.fileStorageShareName, cfg.fileStorageDir, fileName, dirPath + "/" + fileName)
                fileHandle.close()
                
    
    def personaImagesTest(session):
        print("Outputs results")
        
        #Does a table scan for persona summary rows in the persona table
        results = session.execute('SELECT * FROM ' + personaTableName)
        query_persona_name=""
        for result in results:
            query_persona_name = result.persona_name
            break #Leaves the loop, effectively selecting the first element
        
        #Query to find the predicted images associated with that persona
        rawEdges = session.execute('SELECT pred_assoc_image_id FROM ' + personaEdgeTableName + ' WHERE persona_name=%s', (query_persona_name,))
        filteredEdges = list(filter(lambda x: x.pred_assoc_image_id != '', rawEdges))
        predicted_image_bytes = list(map(lambda x: x.pred_assoc_image_id, filteredEdges))
        
            
    def relabelEdgeTest(session):
        '''
        Series of the database operations designed to demonstrate how a user viewing the raw images associated with a person can label them
        '''
        raw_edges = list(session.execute('SELECT persona_name, assoc_image_id, label_assoc_flag, pred_assoc_flag FROM ' + personaEdgeTableName + ' WHERE persona_name=%s', ("Roosevelt",)))
        print("Original Edges")
        print(raw_edges)

        print("Relabeled Edges")
        filtered_edges = list(filter(lambda x: x.pred_assoc_flag == True, raw_edges))
        edges_to_relabel = random.sample(filtered_edges, 1)
        print(edges_to_relabel)
        relabelQuery = session.prepare("UPDATE " + personaEdgeTableName + " SET label_assoc_flag=?, pred_assoc_flag=? WHERE persona_name=? AND assoc_image_id=?")
        for edges_to_relabel in edges_to_relabel:
            session.execute(relabelQuery, (True, False, edges_to_relabel.persona_name, edges_to_relabel.assoc_image_id))
            
        print("Finished edges")
        raw_edges = session.execute('SELECT persona_name, assoc_image_id, label_assoc_flag, pred_assoc_flag FROM ' + personaEdgeTableName + ' WHERE persona_name=%s', ("Roosevelt",))
        print(list(raw_edges))
    
    testGenerateTables(session)
    relabelEdgeTest(session)
    
    
    session.shutdown()
    #Note that pictures converted to bytes vomit for unknown reasons
    #insert_data = session.prepare("INSERT INTO personas (personaId, picture) VALUES (?,?)")
    #session.execute(insert_data, [1,booBytes])
    '''
    results = session.execute('SELECT personaId, picture from personas WHERE personaId in (1, 2)')
    for result in results:
        print(result.personaid)
    '''
