import fitz
from azure.ai.documentintelligence.models import DocumentAnalysisFeature
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from pydantic import BaseModel
from typing import Optional, List
from io import BytesIO
import os
from dotenv import load_dotenv
load_dotenv()


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


doc_client = DocumentIntelligenceClient(
    endpoint=smart_doc_endpoint, credential=AzureKeyCredential(smart_doc_key))

# extrai todo o conteúdo do pdf
def extract_all_content(file_bytes: bytes, filename: str) -> List[ContentBlock]:
    # verifica se as credenciais estão configuradas
    if not smart_doc_endpoint or not smart_doc_key:
        raise ValueError("Le impostazioni di Azure Document Intelligence non sono configurate.")

    # analisa o layout e o texto com o azure document intelligence
    print(f"Analisi del layout iniziale per {filename}...")
    try:
        poller = doc_client.begin_analyze_document(
            "prebuilt-layout",
            BytesIO(file_bytes),
            features=[DocumentAnalysisFeature.OCR_HIGH_RESOLUTION]
        )
        result = poller.result()
    except Exception as e:
        print(f"[ERRO] Azure Document Intelligence falhou: {e}")
        result = None

    # lista onde serão armazenados os blocos extraídos
    content_blocks: List[ContentBlock] = []
    chunk_index_counter = 0

    
    # extrai o conteúdo textual completo
    azure_text_pages = set()
    if result and result.paragraphs:
        for paragraph in result.paragraphs:
            if paragraph.bounding_regions:
                page_number = paragraph.bounding_regions[0].page_number
                if paragraph.content.strip():
                    content_blocks.append(
                        ContentBlock(
                            id=chunk_index_counter,
                            type='text',
                            content=paragraph.content.strip(),
                            page_number=page_number
                        )
                    )
                    azure_text_pages.add(page_number)
                    chunk_index_counter += 1

    # Se ainda assim não encontrou texto, mostra aviso
    try:
        pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            text = page.get_text("text")
            if text.strip() and (page_num + 1) not in azure_text_pages:
                content_blocks.append(ContentBlock(
                    id=chunk_index_counter,
                    type='text',
                    content=text.strip(),
                    page_number=page_num + 1
                ))
                chunk_index_counter += 1
        pdf_document.close()
    except Exception as e:
        print(f"[ERRO] Falha na extração de texto com PyMuPDF: {e}")

    # extrai tabelas e converte para texto
    if result.tables:
        for table in result.tables:
            table_text = "\n".join(
                [f"Tabella a pagina {table.bounding_regions[0].page_number} (Contenuto strutturato):\n" + table.content])

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
