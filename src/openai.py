from dotenv import load_dotenv
load_dotenv()
from openai import AzureOpenAI
import os
import tiktoken

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version = "2024-12-01-preview"
)

embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
 
#funcao responsavel por contar os tokens
def count_tokens(messages: list):
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:
        return 0
    token_count = 0
    for message in messages:
        #adiciona 6 tokens por mensagem
        token_count += 6
        for key, value in message.items():
            token_count += len(encoding.encode(value))

    #adiciona 2 tokens para a sequencia de conclusao        
    return token_count + 2


# gera vetores embedding para um pedaço de texto (chunk).Verifica se o input é uma string válida e não vazia.
def get_embedding(text: str):
    #verifica se o chunk é uma string
    if not isinstance(text, str):
        print(
            f"embedding non valido.")
        return None

    #verifica se o chunk está vazio ou contém apenas espaços em branco
    if not text.strip():
        print("input vuoto o spazi vuoti.")
        return None
    try:
        response = client.embeddings.create(
            model=embedding_deployment,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(
            f"Errore nella creazione dell'embedding", e)
        return None

#gera respostas do gpt baseadas no contexto recuperado do azure search
def chat_with_context(context: str, question: str, history: list):
   # mensagem do Sistema (contém o Contexto RAG)
    system_message = {
        "role": "system", 
        "content": (
            f"You are a RAG assistant. Respond ONLY based on the following context. "
            f"If the answer is not explicitly in context, "
            f"you MUST respond that the information was not found in the document. "
            f"Use the conversation chronology to maintain the contest:\n\n{context}"
        )
    }

    #constrói a lista de mensagens completa sistema + histórico + pergunta Atual
    messages = [system_message] + history

    #adiciona a pergunta atual
    messages.append({"role": "user", "content": question})

    # envia para o azure openai
    response = client.chat.completions.create(
        model=deployment,
        messages=messages
    )
    return response.choices[0].message.content