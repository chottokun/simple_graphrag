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
