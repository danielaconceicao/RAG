from dotenv import load_dotenv
load_dotenv()
from src import smart_doc, openai, search_service, blob_logs
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from src.blob_storage import container_client
import uuid
from typing import Optional


MAX_CONTEXT_TOKENS = 6000 

app = FastAPI(title="RAG Chatbot")
search_service.create_vector_index()

#carrega o pdf
# nome do blob no Azure
blob_name = "BRAZILbrochurev2.pdf"
blob_client = container_client.get_blob_client(blob_name)

# lê bytes direto do Azure
pdf_bytes = blob_client.download_blob().readall()

# extrai texto
pdf_text = smart_doc.extract_text(pdf_bytes)
if isinstance(pdf_text, list):
    pdf_text = " ".join(pdf_text)
elif isinstance(pdf_text, bytes):
    pdf_text = pdf_text.decode("utf-8", errors="ignore")

# Verifica tipo final
""" print("Tipo de pdf_text:", type(pdf_text))
print("Primeiros 200 caracteres:", pdf_text[:200]) """

# corta para evitar exceder limite
chunks = [pdf_text[i:i+7000] for i in range(0, len(pdf_text), 7000)] 

# cria embeddings para cada chunk e filtra os inválidos
embedding_vectors = [
    e for e in (openai.get_embedding(chunk) for chunk in chunks) 
    if e is not None
]

if len(embedding_vectors) == 0:
    raise ValueError("Non è stato generato alcun embedding valido. Controllare il PDF e la funzione get_embedding.")

#calcula o vetor médio de todos os embenddings validos
avg_embedding = np.mean(embedding_vectors, axis=0).tolist()

#indexa no azure search
embedding_vectors = [e for e in (openai.get_embedding(chunk) for chunk in chunks) if e is not None]
avg_embedding = np.mean(embedding_vectors, axis=0).tolist()

#cria documento para upload
doc = {
    "id": blob_name.replace('.', '-'),
    "content": pdf_text,
    "contentVector": avg_embedding
}

#envia para o Azure Search
search_service.upload_documents([doc])

# defina um modelo Pydantic para a entrada
class Question(BaseModel):
    question: str
    session_id: Optional[str] = None 

#recebe uma pergunta e busca trechos relevantes e responde
@app.post("/chat")
async def chat(request: Question):
    # acessa a pergunta atraves do objeto
    question = request.question

    #carrega ou cria uma sessao
    session_id = request.session_id if request.session_id else str(uuid.uuid4())
    history = blob_logs.load_session_history(session_id)

    # busca trechos mais relevantes do pdf
    context_docs = search_service.search_hibryd(question)
    context = " ".join(context_docs)

    #prepara a mensagem do sistema para contagem de tokens
    system_message = {
        "role": "system", 
        "content": (
            f"You are a RAG assistant. Respond ONLY based on the following context. "
            f"If the answer is not explicitly in context, "
            f"you MUST respond that the information was not found in the document. "
            f"Use the conversation chronology to maintain the contest:\n\n{context}"
        )
    }

    # calcula tokens do sistema + tokens do histórico
    current_tokens = openai.count_tokens([system_message] + history)

    if current_tokens >= MAX_CONTEXT_TOKENS:
        
        new_session_id = str(uuid.uuid4())
        limit_message = (
            "Limite de contexto (tokens) alcançado. "
            "Por favor, inicie um novo chat. "
            "Seu histórico foi salvo."
        )
        
        # o bot apenas responde a mensagem de limite.
        return {
            "answer": limit_message,
            "session_id": new_session_id # retorna um novo id para forçar o reinício
        }

    # gera a resposta com o histórico, só ocorre se o limite nao foi atingido
    answer = openai.chat_with_context(context, question, history)

     #salva historico: atualiza o histórico
    new_user_message = {"role": "user", "content": question}
    new_assistant_message = {"role": "assistant", "content": answer}

    history.append(new_user_message)
    history.append(new_assistant_message)

    # salva a resposta no blob logs
    blob_logs.save_session_and_log(session_id, history)
    return {
        "answer": answer,
        "session_id": session_id
    }

#lista pdfs no indice
@app.get("/list_pdfs")
async def list_pdfs():
    return {"pdfs": search_service.list_documents()}