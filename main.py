from dotenv import load_dotenv
load_dotenv()
from src import smart_doc, openai, search_service, blob_logs
from fastapi import FastAPI, Form
import numpy as np

app = FastAPI(title="RAG Chatbot")
search_service.create_vector_index()

#carrega o pdf
pdf_path = "data/BRAZILbrochurev2.pdf"

with open(pdf_path, "rb") as f:
    pdf_bytes = f.read()

# extrai texto
pdf_text = smart_doc.extract_text(pdf_bytes)

# garante que o texto esteja no formato correto
if isinstance(pdf_text, list):
    pdf_text = " ".join(pdf_text)
elif isinstance(pdf_text, bytes):
    pdf_text = pdf_text.decode("utf-8", errors="ignore")

# Verifica tipo final
print("Tipo de pdf_text:", type(pdf_text))
print("Primeiros 200 caracteres:", pdf_text[:200])

# corta para evitar exceder limite
chunks = [pdf_text[i:i+7000] for i in range(0, len(pdf_text), 7000)] 

#cria embeddings para cada chunk
embedding_vectors = [openai.get_embedding(chunk) for chunk in chunks]

#filtra chunks inválidos
embedding_vectors = [e for e in embedding_vectors if e is not None]

if len(embedding_vectors) == 0:
    raise ValueError("Non è stato generato alcun embedding valido. Controllare il PDF e la funzione get_embedding.")

#calcula o vetor médio
avg_embedding = np.mean(embedding_vectors, axis=0).tolist()

#indexa no azure search
embedding_vectors = [e for e in (openai.get_embedding(chunk) for chunk in chunks) if e is not None]
avg_embedding = np.mean(embedding_vectors, axis=0).tolist()

#cria documento para upload
doc = {
    "id": "BRAZILbrochurev2.pdf",
    "content": pdf_text,
    "contentVector": avg_embedding
}

#envia para o Azure Search
search_service.upload_documents([doc])

#recebe uma pergunta e busca trechos relevantes e responde
@app.post("/chat")
async def chat(question: str = Form(...)):
    # busca trechos mais relevantes do pdf
    context_docs = search_service.search_semantic(question)
    context = " ".join(context_docs)

    # gera resposta do gpt usando apenas o contexto do pdf
    answer = openai.chat_with_context(context, question)

    # salva a resposta no Blob Storage
    blob_logs.upload_chat_response(question, answer)
    return {"answer": answer}

#lista pdfs no indice
@app.get("/list_pdfs")
async def list_pdfs():
    return {"pdfs": search_service.list_documents()}
