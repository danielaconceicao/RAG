from dotenv import load_dotenv
load_dotenv()

from azure.storage.blob import BlobServiceClient
import os

# --- CONFIGURAÇÃO INICIAL ---
connect_str = os.getenv("AZURE_BLOB_CONNECT_STR")
container_name = os.getenv("AZURE_BLOB_CONTAINER")

blob_service_client = BlobServiceClient.from_connection_string(connect_str)
container_client = blob_service_client.get_container_client(container_name)


# --- FUNÇÃO PARA UPLOAD ---
def upload_pdf(file_path: str):
    """
    Envia um arquivo PDF local para o container no Azure Blob Storage.
    Retorna a URL pública do arquivo.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    blob_name = os.path.basename(file_path)
    blob_client = container_client.get_blob_client(blob_name)

    with open(file_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)

    print(f"Upload concluído: {blob_name}")
    print(f"URL do blob: {blob_client.url}")
    return blob_client.url


# --- FUNÇÃO PARA LISTAR PDFs ---
def list_pdfs():
    """
    Retorna uma lista com os nomes dos arquivos no container.
    """
    blobs = [b.name for b in container_client.list_blobs()]
    print("📂 PDFs no container:", blobs)
    return blobs

