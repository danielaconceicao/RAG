from azure.storage.blob import BlobServiceClient
import os, json
from datetime import datetime, timezone

blob_connection_string = os.getenv("AZURE_BLOB_CONNECTION_STR")
blob_service_client = BlobServiceClient.from_connection_string(blob_connection_string)

# container onde você quer guardar as respostas do chatbot
log_container_name = os.getenv("AZURE_BLOB_LOGS_CONTAINER")

def upload_chat_response(question: str, answer: str):
    # salva a pergunta e a resposta em formato json dentro do container chatbotlogs
    container_client = blob_service_client.get_container_client(log_container_name)
    
    # cria o json com timestamp
    log_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "answer": answer
    }

    blob_name = f"chat_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    blob_client = container_client.get_blob_client(blob_name)

    blob_client.upload_blob(json.dumps(log_data, indent=2), overwrite=True)
    print(f"Risposta salvata in Blob Storage come {blob_name}")
