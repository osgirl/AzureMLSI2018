import os, uuid, sys, configparser
from azure.storage.blob import BlockBlobService, PublicAccess

config = configparser.ConfigParser()
config.read('config.ini')

# Create the BlockBlockService that is used to call the Blob service for the storage account
block_blob_service = BlockBlobService(account_name=config['DEFAULT']['storage_account_name'], account_key=config['DEFAULT']['storage_account_key'], endpoint_suffix="core.usgovcloudapi.net")

# Create a container called 'images'.
container_name = 'images'
block_blob_service.create_container(container_name)
print('created container')

# Set the permission so the blobs are public.
block_blob_service.set_container_acl(container_name, public_access=PublicAccess.Container)
print('permissions set')

path = '/Users/ephraimsalhanick/Desktop/AzureMLSI2018/Trump_Images'

for filename in os.listdir(path):

	# Upload the file to storage
	block_blob_service.create_blob_from_path(container_name, filename, path+'/'+filename)
	print('uploaded: ' + filename)
