from dotenv import load_dotenv
load_dotenv()

import os
from io import BytesIO
from typing import Optional, List
from pydantic import BaseModel
# --- MUDANÇA 1: ATUALIZAÇÃO DO SDK ---
# Substitui 'azure.ai.formrecognizer' por 'azure.ai.documentintelligence'
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import DocumentAnalysisFeature
import fitz 


smart_doc_endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
smart_doc_key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")


class ContentBlock(BaseModel):
    id: int
    type: str 
    content: str 
    image_bytes: Optional[bytes] = None
    page_number: int
    bounding_box: Optional[List[float]] = None 


doc_client = DocumentIntelligenceClient(endpoint=smart_doc_endpoint, credential=AzureKeyCredential(smart_doc_key))

#usa o document intelligence para analisar o pdf/texto e metadados e o pymupdf para extrair os bytes binários das imagens identificadas. 
def extract_all_content(file_bytes: bytes, filename: str) -> List[ContentBlock]:

    if not smart_doc_endpoint or not smart_doc_key:
        raise ValueError("Le impostazioni di Azure Document Intelligence non sono configurate.")

    # 1. Análise do Documento usando o modelo 'prebuilt-layout'
    print(f"Analisi del layout iniziale per {filename}...")
    poller = doc_client.begin_analyze_document(
        "prebuilt-layout", 
        BytesIO(file_bytes),
        #usamos ocr_high_resolution para melhor precisão em elementos como gráficos
        features=[DocumentAnalysisFeature.OCR_HIGH_RESOLUTION] 
    )
    result = poller.result()
    print("Análise de layout concluída.")

    content_blocks: List[ContentBlock] = []
    chunk_index_counter = 0
    
    #processa o texto conteúdo completo
    if result.content:
        content_blocks.append(
            ContentBlock(
                id=chunk_index_counter, 
                type='text', 
                content=result.content, 
                page_number=1 
            )
        )
        chunk_index_counter += 1

    #processa Tabelas converte para texto para a RAG
    if result.tables:
        for table in result.tables:
            table_text = "\n".join([f"Tabela na página {table.bounding_regions[0].page_number} (Conteúdo estruturado):\n" + table.content])
            
            content_blocks.append(
                ContentBlock(
                    id=chunk_index_counter, 
                    type='table', 
                    content=table_text,
                    page_number=table.bounding_regions[0].page_number,
                )
            )
            chunk_index_counter += 1
            
    #processa Figuras gráficos/imagens e usa PyMuPDF para extrair os bytes inicializando o pymupdf
    try:
        pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        print(f"Erro ao carregar PDF com PyMuPDF: {e}")
        #retorna apenas o texto e tabelas rag parcial
        return content_blocks 


    if result.figures:
        for i, figure in enumerate(result.figures):
            #pega a página da figura
            region = figure.bounding_regions[0]
            page_number = region.page_number
            
            # --- TÉCNICA MAIS SEGURA: EXTRAIR TODAS AS IMAGENS DA PÁGINA ---
            # Nota: O page_number do DI é 1-indexado, o fitz é 0-indexado
            pdf_page = pdf_document.load_page(page_number - 1) 
            images_on_page = pdf_page.get_images(full=True)
            
            if images_on_page:
                xref = images_on_page[0][0] 
                image_info = pdf_document.extract_image(xref)
                image_bytes = image_info["image"] 
                image_ext = image_info["ext"]
                
                initial_content = figure.caption.content if figure.caption else f"Figura {i+1} na página {page_number}"
                
                content_blocks.append(
                    ContentBlock(
                        id=chunk_index_counter, 
                        type='image', 
                        content=initial_content,
                        bytes=image_bytes,
                        page_number=page_number,
                        #adicione a extensão aqui para facilitar o image_captioning
                        bounding_box=[page_number, image_ext] 
                    )
                )
                chunk_index_counter += 1
                
    pdf_document.close()
    
    #retorna a lista completa de texto, tabelas e imagens binárias/metadados
    return content_blocks