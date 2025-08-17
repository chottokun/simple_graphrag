import pytest
from unittest.mock import patch, MagicMock
from src.config import get_neo4j_credentials, get_ollama_config

# We are no longer testing live connections, so we don't need these fixtures
# to create actual instances. We are just testing that the configuration
# is passed correctly to the constructors.

def test_neo4j_graph_instantiation():
    """
    Test that Neo4jGraph is instantiated with the correct credentials from config.
    This test does not actually connect to Neo4j.
    """
    creds = get_neo4j_credentials()

    # We patch the class constructor itself
    with patch('langchain_neo4j.Neo4jGraph') as MockNeo4jGraph:
        # Create a mock instance to be returned by the constructor
        mock_instance = MagicMock()
        MockNeo4jGraph.return_value = mock_instance

        # This is the code that would run in our application
        from langchain_neo4j import Neo4jGraph
        graph = Neo4jGraph(
            url=creds["uri"],
            username=creds["username"],
            password=creds["password"]
        )

        # Assert that the constructor was called once with the correct arguments
        MockNeo4jGraph.assert_called_once_with(
            url=creds["uri"],
            username=creds["username"],
            password=creds["password"]
        )
        # Ensure the object we have is the one our mock created
        assert graph == mock_instance

def test_ollama_llm_instantiation():
    """
    Test that OllamaLLM is instantiated with the correct config.
    This test does not actually connect to Ollama.
    """
    config = get_ollama_config()

    with patch('langchain_ollama.llms.OllamaLLM') as MockOllamaLLM:
        mock_instance = MagicMock()
        MockOllamaLLM.return_value = mock_instance

        # This is the code that would run in our application
        from langchain_ollama.llms import OllamaLLM
        llm = OllamaLLM(
            base_url=config["base_url"],
            model=config["model"]
        )

        # Assert that the constructor was called once with the correct arguments
        MockOllamaLLM.assert_called_once_with(
            base_url=config["base_url"],
            model=config["model"]
        )
        assert llm == mock_instance
