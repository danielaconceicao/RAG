from openai import AzureOpenAI
import os

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-02-15-preview"
)

embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT") 

def get_embedding(text: str):
    response = client.embeddings.create(
        model=embedding_deployment,
        input=text
    )
    return response.data[0].embedding

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
