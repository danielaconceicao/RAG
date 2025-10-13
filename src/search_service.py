from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchFieldDataType, SearchableField,
    VectorSearch, VectorSearchProfile, HnswParameters,
    HnswAlgorithmConfiguration, SearchField
)
import os

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
                stored=True,
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

#envia embeddings ao indice
def upload_documents(docs):
    search_client.upload_documents(docs)
    print("Documenti inviati all'indice.")

#busca trechos mais relevantes no indice
def search_semantic(query: str):
    results = search_client.search(search_text=query, top=3, select=["content", "contentVector"])
    return [r["content"] for r in results]

#busca textual simples 
def search_textual(query: str):
    results = search_client.search(search_text=query, top=3, select=["content"])
    return [r["content"] for r in results]

#busca hibrida combinacao de semantica mais textual
def search_hibryd(query: str):
    #busca vetorial
    semantic_results = search_semantic(query)
    
    #busca textual
    textual_results= search_textual(query)

    #combiana e remove duplicados mantendo a ordem
    combined=[]
    for r in semantic_results + textual_results:
        if r not in combined:
            combined.append(r)
            
    #retorna os 5 mais relevantes
    return combined[:5]