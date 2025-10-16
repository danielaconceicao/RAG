from dotenv import load_dotenv
load_dotenv()
from typing import Optional
import uuid
from src.blob_storage import container_client
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI
from src import smart_doc, openai, search_service, blob_logs, image_captioning

# limite de tokens que o modelo pode receber de uma vez
MAX_CONTEXT_TOKENS = 8000

app = FastAPI(title="RAG Chatbot")

# garante que o índice vetorial já exista no azure search
search_service.create_vector_index()

# carrega o pdf nome do pdf dentro do container pdfs
blob_name = "covid19.pdf"
blob_client = container_client.get_blob_client(blob_name)

# baixa o arquivo diretamente da nuvem bytes
pdf_bytes = blob_client.download_blob().readall()

# extrai o conteúdo estruturado texto, metadados de imagens/tabelas
print("Iniciando extração estruturada do PDF...")
content_blocks = smart_doc.extract_all_content(pdf_bytes, blob_name)

# cria um array de embeddings para cada chunk
docs_to_upload = []

# id base para o documento original
source_doc_id = blob_name.replace('.', '-')

# contador incremental para os pedaços
chunk_index_counter = 0

# itera sobre cada bloco para criar embeddings
for block in content_blocks:
    if block.type == 'text':
        text_content = block.content

        # divide o texto em pedaços menores para embeddings
        chunks = [text_content[i:i+1000] for i in range(0, len(text_content), 1000)]

        for chunk in chunks:
            chunk_embedding = openai.get_embedding(chunk)
            if chunk_embedding:
                doc = {
                    "id": f"{source_doc_id}-{chunk_index_counter}",
                    "content": chunk,
                    "contentVector": chunk_embedding,
                }
                docs_to_upload.append(doc)
                chunk_index_counter += 1

    elif block.type == 'image' and block.image_bytes:
        # gera legenda da imagem
        caption_text = image_captioning.generate_caption_for_rag(
            block.image_bytes, source_doc_id, block.page_number
        )

        if caption_text:
            caption_embedding = openai.get_embedding(caption_text)
            if caption_embedding:
                doc = {
                    "id": f"{source_doc_id}-img-{chunk_index_counter}",
                    "content": caption_text,
                    "contentVector": caption_embedding,
                }
                docs_to_upload.append(doc)
                chunk_index_counter += 1

# se nenhum embedding foi criado, lança erro
if len(docs_to_upload) == 0:
    raise ValueError(
        "Nenhum embedding válido foi gerado. Verifique o PDF e a função get_embedding."
    )

# envia todos os embeddings gerados de texto e imagens e salva no azure search pronta para a ricerca sem.
search_service.upload_documents(docs_to_upload)
""" print(f"{len(docs_to_upload)} blocos enviados para o índice com sucesso!") """

# define modelo Pydantic para a entrada
class Question(BaseModel):
    question: str
    session_id: Optional[str] = None

@app.post("/chat")
async def chat(request: Question):
    # recebe a pergunta do usuario
    question = request.question
    session_id = request.session_id or str(uuid.uuid4())
    # carrega a cronologia 
    history = blob_logs.load_session_history(session_id)
    # recupera os documentos mais relevantes
    context_docs = search_service.search_hibryd(question)
    context = " ".join(context_docs)

    system_message = {
        "role": "system",
        "content": (
            "CRITICAL RULE: If the user message is a simple greeting in ANY language, respond with a polite greeting and short question. "
            "Ignore RAG instructions for this turn.\n\n"
            f"You are a RAG assistant. Respond ONLY based on the following context. "
            f"If the answer is not explicitly in context, respond: 'The information was not found in the document.'\n\n"
            f"Use conversation chronology:\n\n{context}"
        )
    }

    current_tokens = openai.count_tokens([system_message] + history)
    if current_tokens >= MAX_CONTEXT_TOKENS:
        new_session_id = str(uuid.uuid4())
        return {
            "answer": "Limite de contexto alcançado. Inicie um novo chat. Histórico salvo.",
            "session_id": new_session_id
        }

    # envia o contexto e todo o resto para chat gerar a resposta
    answer = openai.chat_with_context(context, question, history)
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    # salva o log da sessao mesmo nao tendo um banco de dados
    blob_logs.save_session_and_log(session_id, history)

    return {"answer": answer, "session_id": session_id}

@app.get("/list_pdfs")
async def list_pdfs():
    return {"pdfs": search_service.list_documents()}
