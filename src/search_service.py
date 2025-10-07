from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchFieldDataType, SearchableField,
    VectorSearch, VectorSearchAlgorithmConfiguration, HnswParameters, VectorSearchProfile
)
import os

search_endpoint = os.getenv("AZURE_AISEARCH_ENDPOINT")
search_key = os.getenv("AZURE_AISEARCH_KEY")
index_name = os.getenv("AZURE_INDEX_NAME")

search_client = SearchClient(search_endpoint, index_name, AzureKeyCredential(search_key))
index_client = SearchIndexClient(search_endpoint, AzureKeyCredential(search_key))

def create_vector_index():
    """Cria o índice vetorial se ainda não existir."""
    try:
        index_client.get_index(index_name)
        print("✅ Índice já existe.")
        return
    except:
        pass

    index = SearchIndex(
        name=index_name,
        fields=[
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SimpleField(name="contentVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single))
        ],
        vector_search=VectorSearch(
            algorithm_configurations=[
                VectorSearchAlgorithmConfiguration(
                    name="vectorConfig",
                    kind="hnsw",
                    parameters=HnswParameters(metric="cosine")
                )
            ],
            profiles=[
                VectorSearchProfile(
                    name="vectorProfile",
                    algorithm_configuration_name="vectorConfig"
                )
            ]
        )
    )
    index_client.create_index(index)
    print("✅ Índice criado com sucesso!")

def upload_documents(docs):
    search_client.upload_documents(docs)
    print("📚 Documentos enviados ao índice.")

def search_semantic(query: str):
    results = search_client.search(search_text=query, top=3)
    return [r["content"] for r in results]
