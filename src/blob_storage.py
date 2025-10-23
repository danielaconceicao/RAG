from dotenv import load_dotenv
load_dotenv()
import os
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceExistsError


# parte responsavel por controlar tudo da storage
blob_connection = os.getenv("AZURE_BLOB_CONNECT_STR")
blob_container = os.getenv("AZURE_BLOB_CONTAINER")

blob_service_client = BlobServiceClient.from_connection_string(blob_connection)
# tenta criar o container se ele não existir / tries to create the container if it does not exist
try:
    container_client = blob_service_client.create_container(blob_container)
    print(f"contenitore '{blob_container}' creato con successo o già esistente.")
except ResourceExistsError:
    # se já existir, apenas obtém a referência / if it already exists, just get the reference
    container_client = blob_service_client.get_container_client(blob_container)
    print(f"contenitore '{blob_container}' esiste già. Connessione stabilita.")
except Exception as e:
    print(f"Errore durante la connessione o la creazione del contenitore: {e}")
    # se houver outro erro, o código principal irá falhar / if there is another error, the main code will fail
    raise

# função responsavel por enviar o arquivo pdf do computador para a pasta container no azure / function responsible for sending the pdf file from the computer to the container folder in azure
def upload_pdf(file_name: str, data: bytes):
    blob_client = container_client.get_blob_client(file_name)
    blob_client.upload_blob(data, overwrite=True)
    return f"Uploaded {file_name}"

# função para salvar um chunk no container 'chunk' / function to save a chunk in the 'chunk' container
def upload_chunk(file_name: str, data: bytes):
    try:
        # conecta no container 'chunk', cria se não existir / connect to the 'chunk' container, create if it doesn't exist
        chunk_container_name = "chunk"
        try:
            chunk_container_client = blob_service_client.create_container(chunk_container_name)
        except ResourceExistsError:
            chunk_container_client = blob_service_client.get_container_client(chunk_container_name)
        
        # cria o blob client e faz o upload / creates the client blob and uploads it
        blob_client = chunk_container_client.get_blob_client(file_name)
        blob_client.upload_blob(data, overwrite=True)
        print(f"Chunk salvato: {file_name}")
        return f"Chunk salvato: {file_name}"
    
    except Exception as e:
        print(f"Errore durante il salvataggio del chunk {file_name}: {e}")
        raise


# mostra arquivos que se encontram no container / show files that are in the container
def list_pdfs():
    return [b.name for b in container_client.list_blobs()]