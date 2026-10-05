"""Document/query embeddings through a separate Azure OpenAI deployment."""
from langchain_openai import AzureOpenAIEmbeddings
from config import AZURE_OPENAI_EMBEDDING_DEPLOYMENT, EMBEDDING_MODEL, azure_client_settings


def get_embeddings() -> AzureOpenAIEmbeddings:
    return AzureOpenAIEmbeddings(
        **azure_client_settings(AZURE_OPENAI_EMBEDDING_DEPLOYMENT),
        model=EMBEDDING_MODEL, tiktoken_model_name=EMBEDDING_MODEL,
        request_timeout=60, chunk_size=50,
    )
