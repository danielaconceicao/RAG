from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchFieldDataType, SearchableField,
    VectorSearch, VectorSearchProfile, HnswParameters,
    HnswAlgorithmConfiguration, SearchField
)
import os
from src.openai import get_embedding
from azure.search.documents.models import VectorizedQuery
import uuid

search_endpoint = os.getenv("AZURE_AISEARCH_ENDPOINT")
search_key = os.getenv("AZURE_AISEARCH_KEY")
index_name = os.getenv("AZURE_AISEARCH_INDEX_NAME")

search_client = SearchClient(search_endpoint, index_name, AzureKeyCredential(search_key))

index_client = SearchIndexClient(search_endpoint, AzureKeyCredential(search_key))


# cria o índice vetorial se ainda não existir.
def create_vector_index():
    try:
        index_client.get_index(index_name)
        print("L'indice gia esiste.")
        return
    except:
        pass

    VECTOR_DIMENSIONS = 1536
    VECTOR_PROFILE_NAME = "vectorProfile"
    VECTOR_ALGORITHM_NAME = "hnsw-config"

    index = SearchIndex(
        name=index_name,
        fields=[
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="content", type=SearchFieldDataType.String),

            SearchField(
                name="contentVector",
                type=SearchFieldDataType.Collection(
                    SearchFieldDataType.Single),
                searchable=True,
                retrievable=True,
                vector_search_dimensions=VECTOR_DIMENSIONS,
                vector_search_profile_name=VECTOR_PROFILE_NAME
            )


        ],

        vector_search=VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name=VECTOR_ALGORITHM_NAME,
                    parameters=HnswParameters(metric="cosine")
                )
            ],
            profiles=[
                VectorSearchProfile(
                    name=VECTOR_PROFILE_NAME,
                    algorithm_configuration_name=VECTOR_ALGORITHM_NAME
                )
            ]
        )
    )
    
    index_client.create_index(index)
    print("Indice creato con successo!")

# envia embeddings ao indice
def upload_documents(chunks):
    docs = []
    for chunk in chunks:
        content = chunk.get("content") if isinstance(chunk, dict) else str(chunk)

        # gera o embedding
        embedding = get_embedding(content)
        if not embedding:
            continue

        doc = {
            "id": str(uuid.uuid4()),
            "content": content,
            "contentVector": embedding,  
        }
        docs.append(doc)

    search_client.upload_documents(docs)

# busca trechos mais relevantes no indice
def search_semantic(query: str):
    query_vector = get_embedding(query)

    # crie a consulta vetorial
    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=20,
        fields="contentVector"
    )

    results = search_client.search(
        search_text="",  # texto vazio porque estamos usando vetor
        vector_queries=[vector_query],
        select=["content"]
    )

    # Retorne os resultados como lista
    return [r["content"] for r in results if "content" in r]


#busca textual simples 
def search_textual(query: str):
    results = search_client.search(search_text=query, top=5, select=["content"])
    return [r["content"] for r in results]

#busca hibrida combinacao de semantica mais textual
def search_hibryd(query: str):
    vector_results = search_semantic(query)
    textual_results = search_client.search(
        search_text=query,
        top=5,
        select=["content"]
    )
    text_results = [r["content"] for r in textual_results if "content" in r]
    # remove duplicados mantendo a ordem
    combined = []
    for r in vector_results + text_results:
        if r not in combined:
            combined.append(r)
    return combined[:5]