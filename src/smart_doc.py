from dotenv import load_dotenv
load_dotenv()

import os
from io import BytesIO
from typing import Optional, List
from pydantic import BaseModel
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import DocumentAnalysisFeature
import fitz 


smart_doc_endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
smart_doc_key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

#   modelo pydantic para representar um bloco do pdf
class ContentBlock(BaseModel):
    id: int
    type: str 
    content: str 
    image_bytes: Optional[bytes] = None
    page_number: int
    bounding_box: Optional[List[float]] = None 


doc_client = DocumentIntelligenceClient(endpoint=smart_doc_endpoint, credential=AzureKeyCredential(smart_doc_key))

# extrai todo o conteúdo do pdf 
def extract_all_content(file_bytes: bytes, filename: str) -> List[ContentBlock]:
    ## verifica se as credenciais estão configuradas
    if not smart_doc_endpoint or not smart_doc_key:
        raise ValueError("Le impostazioni di Azure Document Intelligence non sono configurate.")

    # analisa o layout e o texto com o azure document intelligence
    print(f"Analisi del layout iniziale per {filename}...")
    poller = doc_client.begin_analyze_document(
        "prebuilt-layout", 
        BytesIO(file_bytes),
        #usamos ocr_high_resolution para melhor precisão em elementos como gráficos
        features=[DocumentAnalysisFeature.OCR_HIGH_RESOLUTION] 
    )
    # espera o resultado completo
    result = poller.result()

    # lista onde serão armazenados os blocos extraídos
    content_blocks: List[ContentBlock] = []
    chunk_index_counter = 0
    
    # extrai o conteúdo textual completo
    if result.content:
        content_blocks.append(
            ContentBlock(
                id=chunk_index_counter, 
                type='text', 
                # todo o texto extraído do pdf
                content=result.content, 
                page_number=1 
            )
        )
        chunk_index_counter += 1

    #extrai tabelas e converte para texto
    if result.tables:
        for table in result.tables:
            table_text = "\n".join([f"Tabella a pagina {table.bounding_regions[0].page_number} (Contenuto strutturato):\n" + table.content])
            
            content_blocks.append(
                ContentBlock(
                    id=chunk_index_counter, 
                    type='table', 
                    content=table_text,
                    page_number=table.bounding_regions[0].page_number,
                )
            )
            chunk_index_counter += 1
            
    # extrai imagens e gráficos com pymupdf
    try:
         # abre o pdf em memória
        pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        print(f"Errore durante il caricamento del PDF con PyMuPDF: {e}")
        return content_blocks

    # percorre cada página e coleta as imagens
    for page_number in range(len(pdf_document)):
        pdf_page = pdf_document.load_page(page_number)
        images_on_page = pdf_page.get_images(full=True)

        for img_index, img_info in enumerate(images_on_page):
            xref = img_info[0]
            image_info = pdf_document.extract_image(xref)
            image_bytes = image_info["image"]
            image_ext = image_info["ext"]

             # cria um bloco para cada imagem
            content_blocks.append(
                ContentBlock(
                    id=chunk_index_counter,
                    type='image',
                    content=f"Imagem {img_index + 1} na página {page_number + 1}",
                    image_bytes=image_bytes,
                    page_number=page_number + 1,
                    bounding_box=None
                )
            )
            chunk_index_counter += 1
                
    pdf_document.close()
    
    # retorna todos os blocos extraídos a lista completa de texto, tabelas e imagens binárias/metadados
    return content_blocks