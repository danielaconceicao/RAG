from dotenv import load_dotenv
load_dotenv()

from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
import os
from io import BytesIO

smart_doc_endpoint = os.getenv("AZURE_SMART_DOCUMENT_ENDPOINT")
smart_doc_key = os.getenv("AZURE_SMART_DOCUMENT_KEY")

""" variavel responsavel por fazer a ponte tra o meu codigo e o azure """
doc_client = DocumentAnalysisClient(
    endpoint=smart_doc_endpoint, credential=AzureKeyCredential(smart_doc_key))

""" funcao responsavel por receber bytes do arquivo pdf e retornar o texto extraido """
def extract_text(file_bytes: bytes) -> str:
    """ enviando documento para ser processado no azure """
    poller = doc_client.begin_analyze_document(
        "prebuilt-read", document=BytesIO(file_bytes)) 
    
    """ contem todas as páginas, linhas e palavras extraídas do pdf """
    result = poller.result()

    text = " ".join(
        [line.content for page in result.pages for line in page.lines])
    return text
