import os
from dotenv import load_dotenv

# This function can be called explicitly at the start of the actual application
# to load the .env file. For tests, this is handled by conftest.py.
def load_app_config():
    """Load environment variables from .env file."""
    load_dotenv()

def get_neo4j_credentials():
    """
    Returns a dictionary of Neo4j credentials from environment variables.
    """
    return {
        "uri": os.getenv("NEO4J_URI"),
        "username": os.getenv("NEO4J_USERNAME"),
        "password": os.getenv("NEO4J_PASSWORD"),
    }

def get_ollama_config():
    """
    Returns a dictionary of Ollama configuration from environment variables.
    """
    return {
        "base_url": os.getenv("OLLAMA_BASE_URL"),
        "model": os.getenv("OLLAMA_MODEL"),
        "embedding_model": os.getenv("OLLAMA_EMBEDDING_MODEL"),
    }

def get_llm_config():
    """
    Returns a dictionary of LLM configuration from environment variables.
    """
    llm_provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    config = {"provider": llm_provider}

    if llm_provider == "ollama":
        config["base_url"] = os.getenv("OLLAMA_BASE_URL")
        config["model"] = os.getenv("OLLAMA_MODEL")
        config["embedding_model"] = os.getenv("OLLAMA_EMBEDDING_MODEL")
    elif llm_provider == "openai":
        config["api_key"] = os.getenv("OPENAI_API_KEY")
        config["model"] = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        config["embedding_model"] = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    elif llm_provider == "azure_openai":
        config["api_key"] = os.getenv("AZURE_OPENAI_API_KEY")
        config["azure_endpoint"] = os.getenv("AZURE_OPENAI_ENDPOINT")
        config["api_version"] = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        config["model"] = os.getenv("AZURE_OPENAI_MODEL") # Deployment name
        config["embedding_model"] = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL") # Deployment name
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {llm_provider}")
    return config

def get_graph_transformer_config():
    """
    Returns a dictionary of graph transformer configuration from environment variables.
    """
    allowed_nodes_str = os.getenv("GRAPH_ALLOWED_NODES")
    allowed_relationships_str = os.getenv("GRAPH_ALLOWED_RELATIONSHIPS")

    allowed_nodes = [node.strip() for node in allowed_nodes_str.split(',')] if allowed_nodes_str else None
    allowed_relationships = [rel.strip() for rel in allowed_relationships_str.split(',')] if allowed_relationships_str else None

    return {
        "allowed_nodes": allowed_nodes,
        "allowed_relationships": allowed_relationships,
    }
