from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError
import os, json
from datetime import datetime, timezone

blob_connection_string = os.getenv("AZURE_BLOB_CONNECT_STR")
blob_service_client = BlobServiceClient.from_connection_string(blob_connection_string)

# container onde você quer guardar as respostas do chatbot
log_container_name = os.getenv("AZURE_BLOB_LOGS_CONTAINER")

#salva o histórico completo da sessão como o arquivo de memória e loga cada interação
def save_session_and_log(session_id: str, history: list):
    container_client = blob_service_client.get_container_client(log_container_name)
    
    # cria o nome do blob de memoria, usando o id da sessão
    blob_name = f"session_memory/{session_id}.json"
    blob_client = container_client.get_blob_client(blob_name)
    
    # prepara o objeto completo para salvar
    log_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "history": history
    }

    # salva o estado atual da sessão isto garante a persistência
    try:
        blob_client.upload_blob(json.dumps(log_data, indent=2), overwrite=True)
        print(f"Sessione salvata e mantenuta: {blob_name}")
    except Exception as e:
        print(f"Errore durante il salvataggio della sessione in Blob Storage: {e}")


def load_session_history(session_id: str) -> list:
    """carrega o histórico de uma sessão existente do blob storage."""
    if not session_id:
        return []
        
    container_client = blob_service_client.get_container_client(log_container_name)
    blob_name = f"session_memory/{session_id}.json"
    blob_client = container_client.get_blob_client(blob_name)
    
    try:
        # tenta baixar o conteúdo do blob
        download_stream = blob_client.download_blob()
        data = json.loads(download_stream.readall().decode('utf-8'))
        
        # retorna apenas o histórico, se existir
        return data.get("history", [])
        
    except ResourceNotFoundError:
        # se o arquivo não existir (nova sessão ou sessão expirada)
        print(f"Sessione {session_id} non trovato. Avvio di una nuova sessione.")
        return []
    except Exception as e:
        print(f"Errore durante il caricamento della sessione {session_id}: {e}")
        return []