from fastapi import FastAPI, Form
from src import smart_doc, openai, search_service, blob_logs

app = FastAPI(title="RAG Chatbot")


pdf_path = "data/O-Alienista.pdf"

with open(pdf_path, "rb") as f:
    pdf_bytes = f.read()

# extrai texto
pdf_text = smart_doc.extract_text(pdf_bytes)

# cria embeddings
embedding = openai.get_embedding(pdf_text)

# indexa no azure search (uma vez)
doc = {
    "id": "O-Alienista.pdf",
    "content": pdf_text,
    "contentVector": embedding
}

search_service.upload_documents([doc])
print(f"PDF '{pdf_path}' indexado com sucesso!")


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


@app.get("/list_pdfs")
async def list_pdfs():
    return {"pdfs": search_service.list_documents()}
