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
    
    # se a imagem estiver vazia, retorna uma string vazia
    if not image_bytes:
        return ""
    
    # converte os bytes da imagem para Base64
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    
    # cria o prompt que será enviado ao modelo
    prompt = (
        "Analyze the visual content of this image, which may be a graph,"
        "a table or a figure. Extract and describe the main information "
        "and relevant data. If it's a graph, mention peaks, "
        "notable trends or values. Your answer will be used as context "
        "for a RAG chatbot. Start with 'Image of {filename} on the page {page_number}: '. "
        "Não gere nada além da descrição do conteúdo visual."
    ).format(filename=filename, page_number=page_number)

    # monta a mensagem no formato aceito pelo modelo multimodal
    # o modelo de visão recebe um array messages, onde cada item pode conter texto e imagens
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        # envia a imagem diretamente como base64, simulando uma URL de imagem
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
        # retorna apenas o texto gerado a legenda
        return response.choices[0].message.content
    except Exception as e:
        print(f"Errore durante la generazione della didascalia per l'immagine: {e}")
        return ""