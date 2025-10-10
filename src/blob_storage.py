from dotenv import load_dotenv
load_dotenv()
import os
from azure.storage.blob import BlobServiceClient


# parte responsavel por controlar tudo da storage
blob_connection = os.getenv("AZURE_BLOB_CONNECT_STR")
blob_container = os.getenv("AZURE_BLOB_CONTAINER")

blob_service_client = BlobServiceClient.from_connection_string(blob_connection)
container_client = blob_service_client.get_container_client(blob_container)

#função responsavel por enviar o arquivo pdf do computador para a pasta container no azure 
def upload_pdf(file_name: str, data: bytes):
    blob_client = container_client.get_blob_client(file_name)
    blob_client.upload_blob(data, overwrite=True)
    return f"Uploaded {file_name}"


#mostra arquivos que se encontram no container 
def list_pdfs():
    return [b.name for b in container_client.list_blobs()]
