from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI, AzureChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from src.config import get_llm_config

def get_llm_and_embeddings() -> (BaseLanguageModel, Embeddings):
    """
    Initializes and returns the appropriate LLM and Embeddings model
    based on the configuration in src/config.py.
    """
    llm_config = get_llm_config()
    provider = llm_config["provider"]

    llm: BaseLanguageModel
    embeddings: Embeddings

    if provider == "ollama":
        llm = ChatOllama(
            base_url=llm_config["base_url"],
            model=llm_config["model"]
        )
        embeddings = OllamaEmbeddings(
            base_url=llm_config["base_url"],
            model=llm_config["embedding_model"]
        )
    elif provider == "openai":
        llm = ChatOpenAI(
            api_key=llm_config["api_key"],
            model=llm_config["model"]
        )
        embeddings = OpenAIEmbeddings(
            api_key=llm_config["api_key"],
            model=llm_config["embedding_model"]
        )
    elif provider == "azure_openai":
        llm = AzureChatOpenAI(
            api_key=llm_config["api_key"],
            azure_endpoint=llm_config["azure_endpoint"],
            api_version=llm_config["api_version"],
            azure_deployment=llm_config["model"]
        )
        embeddings = OpenAIEmbeddings(
            api_key=llm_config["api_key"],
            azure_endpoint=llm_config["azure_endpoint"],
            api_version=llm_config["api_version"],
            azure_deployment=llm_config["embedding_model"]
        )
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")

    return llm, embeddings
