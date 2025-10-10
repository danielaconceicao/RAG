from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError
import os, json
from datetime import datetime, timezone

blob_connection_string = os.getenv("AZURE_BLOB_CONNECT_STR")
blob_service_client = BlobServiceClient.from_connection_string(blob_connection_string)

# container onde você quer guardar as respostas do chatbot
log_container_name = os.getenv("AZURE_BLOB_LOGS_CONTAINER")

def save_session_and_log(session_id: str, history: list):
    """
    Salva o histórico COMPLETO da sessão como o arquivo de 'memória' 
    e loga cada interação.

    No nosso caso, vamos usar a mesma operação para salvar o estado mais recente.
    """
    container_client = blob_service_client.get_container_client(log_container_name)
    
    # 1. Cria o nome do blob de MEMÓRIA, usando o ID da sessão
    # Exemplo: 'session_memory/c5752c03-d64e-4f7f-8d99-8083c27e85c2.json'
    blob_name = f"session_memory/{session_id}.json"
    blob_client = container_client.get_blob_client(blob_name)
    
    # Prepara o objeto completo para salvar
    log_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "history": history
    }

    # Salva o estado atual da sessão (Isto garante a persistência!)
    try:
        blob_client.upload_blob(json.dumps(log_data, indent=2), overwrite=True)
        print(f"✅ Sessão salva e persistida: {blob_name}")
    except Exception as e:
        print(f"❌ Erro ao salvar sessão no Blob Storage: {e}")


def load_session_history(session_id: str) -> list:
    """Carrega o histórico de uma sessão existente do Blob Storage."""
    if not session_id:
        return []
        
    container_client = blob_service_client.get_container_client(log_container_name)
    blob_name = f"session_memory/{session_id}.json"
    blob_client = container_client.get_blob_client(blob_name)
    
    try:
        # Tenta baixar o conteúdo do blob
        download_stream = blob_client.download_blob()
        data = json.loads(download_stream.readall().decode('utf-8'))
        
        # Retorna apenas o histórico, se existir
        return data.get("history", [])
        
    except ResourceNotFoundError:
        # Se o arquivo não existir (nova sessão ou sessão expirada)
        print(f"Sessão {session_id} não encontrada. Iniciando nova sessão.")
        return []
    except Exception as e:
        print(f"❌ Erro ao carregar sessão {session_id}: {e}")
        return []