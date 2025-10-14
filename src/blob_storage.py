from dotenv import load_dotenv
load_dotenv()
import os
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceExistsError


# parte responsavel por controlar tudo da storage
blob_connection = os.getenv("AZURE_BLOB_CONNECT_STR")
blob_container = os.getenv("AZURE_BLOB_CONTAINER")

blob_service_client = BlobServiceClient.from_connection_string(blob_connection)
# tenta criar o container se ele não existir
try:
    container_client = blob_service_client.create_container(blob_container)
    print(f"contenitore '{blob_container}' creato con successo o già esistente.")
except ResourceExistsError:
    # se já existir, apenas obtém a referência
    container_client = blob_service_client.get_container_client(blob_container)
    print(f"contenitore '{blob_container}' esiste già. Connessione stabilita.")
except Exception as e:
    print(f"Errore durante la connessione o la creazione del contenitore: {e}")
    # se houver outro erro, o código principal irá falhar.
    raise

#função responsavel por enviar o arquivo pdf do computador para a pasta container no azure 
def upload_pdf(file_name: str, data: bytes):
    blob_client = container_client.get_blob_client(file_name)
    blob_client.upload_blob(data, overwrite=True)
    return f"Uploaded {file_name}"

# função para salvar um chunk no container 'chunk'
def upload_chunk(file_name: str, data: bytes):
    try:
        # conecta no container 'chunk', cria se não existir
        chunk_container_name = "chunk"
        try:
            chunk_container_client = blob_service_client.create_container(chunk_container_name)
        except ResourceExistsError:
            chunk_container_client = blob_service_client.get_container_client(chunk_container_name)
        
        # cria o blob client e faz o upload
        blob_client = chunk_container_client.get_blob_client(file_name)
        blob_client.upload_blob(data, overwrite=True)
        print(f"Chunk salvato: {file_name}")
        return f"Chunk salvato: {file_name}"
    
    except Exception as e:
        print(f"Errore durante il salvataggio del chunk {file_name}: {e}")
        raise


#mostra arquivos que se encontram no container 
def list_pdfs():
    return [b.name for b in container_client.list_blobs()]