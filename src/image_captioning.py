# src/image_captioning.py

from dotenv import load_dotenv
load_dotenv()
from openai import AzureOpenAI
import os
import base64

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_VISIONIMAGE_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_VISIONIMAGE_ENDPOINT"),
    api_version = "2024-12-01-preview"
)

VISION_DEPLOYMENT = os.getenv("AZURE_OPENAI_VISIONIMAGE_DEPLOYMENT") 

#Usa o modelo de visão para gerar uma descrição/legenda detalhada para uma imagem, focando em dados, tabelas ou gráficos
def generate_caption_for_rag(image_bytes: bytes, filename: str, page_number: int) -> str:
    
    #se nao tiver imagem me retorne nada
    if not image_bytes:
        return ""
    
    #codifica a imagem para Base64 o formato que o LLM Vision aceita na API
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    
    prompt = (
        "Analyze the visual content of this image, which may be a graph,"
        "a table or a figure. Extract and describe the main information "
        "and relevant data. If it's a graph, mention peaks, "
        "notable trends or values. Your answer will be used as context "
        "for a RAG chatbot. Start with 'Image of {filename} on the page {page_number}: '. "
        "Não gere nada além da descrição do conteúdo visual."
    ).format(filename=filename, page_number=page_number)


    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        # Assumimos que o formato é jpeg, mas pode ser png, etc.
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ]

    try:
        response = client.chat.completions.create(
            model=VISION_DEPLOYMENT, 
            messages=messages,
            max_tokens=500 
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Errore durante la generazione della didascalia per l'immagine: {e}")
        return ""