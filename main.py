from dotenv import load_dotenv
load_dotenv()
from typing import Optional
import uuid
from src.blob_storage import container_client, upload_chunk
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI
from src import smart_doc, openai, search_service, blob_logs, image_captioning

# limite de tokens que o modelo pode receber de uma vez
MAX_CONTEXT_TOKENS = 6000

app = FastAPI(title="RAG Chatbot")

# garante que o índice vetorial já exista no azure search
search_service.create_vector_index()

# carrega o pdf nome do pdf dentro do container pdfs
blob_name = "covid19.pdf"
blob_client = container_client.get_blob_client(blob_name)

# baixa o arquivo diretamente da nuvem bytes
pdf_bytes = blob_client.download_blob().readall()

# extrai o conteúdo estruturado texto, metadados de imagens/tabelas
""" print("Avvio dell'estrazione PDF strutturata...") """
content_blocks = smart_doc.extract_all_content(pdf_bytes, blob_name)

# cria um array de embeddings para cada chunk
docs_to_upload = []

# id base para o documento original
source_doc_id = blob_name.replace('.', '-')

# contador incremental para os pedaços
chunk_index_counter = 0

# itera sobre cada chunk para criar um documento embedding separado
for block in content_blocks:
    if block.type == 'text':
        #texto
        text_content = block.content

        # divide o texto em pedaços menores
        chunks = [text_content[i:i+7000] for i in range(0, len(text_content), 7000)]

        for i, chunk in enumerate(chunks):
            # cria embedding do texto com modelo da openai
            chunk_embedding = openai.get_embedding(chunk)

            # cria o documento que será enviado ao azure search
            if chunk_embedding is not None:
                doc = {
                    "id": f"{source_doc_id}-{chunk_index_counter}", 
                    "content": chunk,
                    "contentVector": chunk_embedding,
                }
                docs_to_upload.append(doc)
                chunk_index_counter += 1

    elif block.type == 'image' and block.image_bytes:
        # PROCESSAMENTO DE IMAGEM/TABELA (LEGENDAGEM E EMBEDDING)
        """ print(f"Generazione della didascalia per {block.type} sulla pagina {block.page_number}...") """
        
        # converte a imagem em uma legenda textual descritiva 
        caption_text = image_captioning.generate_caption_for_rag(
             block.image_bytes, source_doc_id, block.page_number 
        )

        if caption_text:
            # gera embedding para a legenda trata como texto
            caption_embedding = openai.get_embedding(caption_text)
            
            if caption_embedding is not None:
                # cria documento para o azure search com a descrição da imagem
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
        "Non è stato generato alcun embedding valido. Controllare il PDF e la funzione get_embedding.")

#  enviar embeddings para o azure search, agora o documento texto + imagens está indexado e pronto para buscas semânticas
search_service.upload_documents(docs_to_upload)


# defina um modelo Pydantic para a entrada
class Question(BaseModel):
    question: str
    session_id: Optional[str] = None

# recebe uma pergunta e busca trechos relevantes e responde
@app.post("/chat")
async def chat(request: Question):
    # pega a pergunta enviada pelo usuário
    question = request.question

    # usa o mesmo session_id se existir, senão cria um novo
    session_id = request.session_id if request.session_id else str(
        uuid.uuid4())
    
    # carrega histórico da conversa
    history = blob_logs.load_session_history(session_id)

    # busca trechos mais relevantes do pdf
    context_docs = search_service.search_hibryd(question)
    context = " ".join(context_docs)

    # mensagem de sistema que define as regras do comportamento do modelo
    system_message = {
        "role": "system",
        "content": (
            f"CRITICAL RULE: If the user message is a simple greeting in ANY language (e.g., 'hi', 'hello', 'ciao', 'bom dia', 'salut', 'konnichiwa', etc.), "
            f"you MUST respond immediately with a polite greeting in the same language and a short question like 'How can I help you today?' (or equivalent), "
            f"and you MUST completely ignore all RAG instructions and context for that turn. This rule has absolute priority.\n\n"

            f"You are a RAG assistant. Your remaining instructions are to respond ONLY based on the following context. "
            f"If the answer is not explicitly in context, "
            f"you MUST respond with the exact phrase: 'The information was not found in the document.'\n\n"

            f"Use the conversation chronology to maintain the context:\n\n{context}"
        )
    }

    # calcula tokens do sistema + tokens do histórico
    current_tokens = openai.count_tokens([system_message] + history)

    # se o limite de tokens for atingido, reinicia a sessão
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
            # retorna um novo id para forçar o reinício
            "session_id": new_session_id  
        }

    # caso contrário, gera a resposta final usando contexto + histórico
    answer = openai.chat_with_context(context, question, history)

    # Atualiza histórico da sessão com as novas mensagens
    new_user_message = {"role": "user", "content": question}
    new_assistant_message = {"role": "assistant", "content": answer}
    history.append(new_user_message)
    history.append(new_assistant_message)

    # salva tudo no azure blob para persistência da conversa
    blob_logs.save_session_and_log(session_id, history)
    return {
        "answer": answer,
        "session_id": session_id
    }

# lista pdfs no indice
@app.get("/list_pdfs")
async def list_pdfs():
    return {"pdfs": search_service.list_documents()}
