#This script deploys the cluster secrets, these are used rather then yaml configs to avoid the complications of
#BASE64 conversions

SECRET_NAME=
REGISTRY_NAME=
YOUR_MAIL=
SERVICE_PRINCIPAL_ID=
YOUR_PASSWORD=

BLOB_SECRET_NAME=
BLOB_STORAGE_ACCOUNT=
BLOB_STORAGE_KEY=

DB_SECRET_NAME=
DB_ACCOUNT=
DB_KEY=

#Azure Container Registry credentials
kubectl create secret docker-registry $SECRET_NAME --docker-server $REGISTRY_NAME --docker-email $YOUR_MAIL --docker-username=$SERVICE_PRINCIPAL_ID --docker-password $YOUR_PASSWORD

#Blob storage secret
kubectl create secret generic $BLOB_SECRET_NAME --from-literal=blob-storage-account=$BLOB_STORAGE_ACCOUNT --from-literal=blob-storage-key=$BLOB_STORAGE_KEY

#
kubectl create secret generic $DB_SECRET_NAME --from-literal=db-account=$DB_ACCOUNT --from-literal=db-key=$DB_KEY
