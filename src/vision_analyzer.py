from dotenv import load_dotenv
load_dotenv()
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from azure.core.credentials import AzureKeyCredential
import os

# carrega credenciais do .env
cv_endpoint = os.getenv("AZURE_CV_ENDPOINT")
cv_key = os.getenv("AZURE_CV_KEY")

# cliente Computer Vision
vision_client = ComputerVisionClient(
    cv_endpoint, cv_key
)

#recebe bytes de uma imagem estraida do pdf e retorna uma descricao textual
def analyze_image(image_bytes: bytes) -> str:
    try:
        result = vision_client.analyze(
            image_data=image_bytes,
            visual_features=[VisualFeatureTypes.CAPTION, VisualFeatureTypes.TAGS]
        )

        description = ""
        if result.caption:
            description += f"Description: {result.caption.text}\n"
        if result.tags:
            tags = ", ".join([t.name for t in result.tags])
            description += f"Tags: {tags}"

        return description.strip() or "Nessuna descrizione trovata."

    except Exception as e:
        print("Errore di analisi dell'immagine:", e)
        return ""
