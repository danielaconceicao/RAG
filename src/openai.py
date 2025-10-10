from dotenv import load_dotenv
load_dotenv()
from openai import AzureOpenAI
import os

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version = "2024-12-01-preview"
)

embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
 


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
def chat_with_context(context: str, question: str):
    prompt = f"Basati solo sul seguente testo:\n\n{context}\n\nDomanda: {question}"
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": "Rispondi solo in base ai documenti caricati."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content
