'''
    This script runs once upon initial instantiation of the container to load the test data.
    The container can be run without running this script. 
'''
from azure.storage.file import FileService
from azure.storage.file import ContentSettings
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from ssl import PROTOCOL_TLSv1_2, CERT_REQUIRED

import ClusterInitializerConfig as cfg

import imageio


if __name__ == '__main__':
    '''
    fileService = FileService(account_name=cfg[fileAccountName], account_key=cfg[fileAccountSecret])

    #Get 
    for file in fileService.list_directories_and_files(cfg[shareName], cfg[dirName]):
        print(file.name)

    '''
    #grabbed my CA.pem from Mozilla https://curl.haxx.se/docs/caextract.html
    ssl_opts = {
        'ca_certs': './cacert.pem',
        'ssl_version': PROTOCOL_TLSv1_2,
        'cert_reqs': CERT_REQUIRED  # Certificates are required and validated
    }

    auth_provider = PlainTextAuthProvider(username=cfg.cosmoDBUsername, password=cfg.cosmoDBSecret)
    cluster = Cluster([cfg.cosmoDBEndpointUri], port = 10350, auth_provider=auth_provider, ssl_options=ssl_opts)
    #session = cluster.connect("<your-keyspace>")
    session = cluster.connect()

    #Creates keyspace from variable name
    print("\nCreating Keyspace")
    session.execute('CREATE KEYSPACE IF NOT EXISTS ' + cfg.cosmoDBKeyspaceName + ' WITH replication = {\'class\': \'NetworkTopologyStrategy\', \'datacenter\' : \'1\' }'); #Formatting is stupid on this string due to the additional curley braces
    session.shutdown()
    
    #Replace with test image of your choice
    booFile = open('./chompette.jpeg', 'rb')
    booBytes = booFile.read(); #Loads as a bython Bytes or bytestring object
    
    session = cluster.connect(cfg.cosmoDBKeyspaceName)
    
    #Create table
    print("\nCreating Table")
    session.execute('DROP TABLE ' + cfg.cosmoDBTableName)
    session.execute('CREATE TABLE IF NOT EXISTS ' + cfg.cosmoDBKeyspaceName + '.' + cfg.cosmoDBTableName + ' (personaId int PRIMARY KEY, picture blob)');

    #Note that pictures converted to bytes vomit for unknown reasons
    insert_data = session.prepare("INSERT INTO personas (personaId, picture) VALUES (?,?)")
    session.execute(insert_data, [1,booBytes])
    
    results = session.execute('SELECT personaId, picture from personas WHERE personaId in (1, 2)')
    for result in results:
        print(result.personaid)
    