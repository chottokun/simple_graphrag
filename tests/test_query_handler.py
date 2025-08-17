import pytest
from unittest.mock import patch, MagicMock

from src.query_handler import QueryHandler
from langchain_neo4j import Neo4jGraph
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import Runnable

@pytest.fixture
def mock_llm():
    """Fixture for a mocked LLM."""
    llm = MagicMock(spec=BaseLanguageModel)
    # Configure the mock to return a string, as a real LLM would.
    llm.invoke.return_value = "Final Answer"
    return llm

@pytest.fixture
def mock_graph():
    """Fixture for a mocked Neo4jGraph."""
    return MagicMock(spec=Neo4jGraph)

@pytest.fixture
def mock_embeddings():
    """Fixture for a mocked Embeddings model."""
    return MagicMock(spec=Embeddings)

def test_retriever_creation(mock_graph, mock_llm, mock_embeddings):
    """
    Tests that the vector retriever is created correctly.
    """
    with patch('src.query_handler.Neo4jVector') as MockNeo4jVector:
        mock_vector_store_instance = MagicMock()
        mock_retriever = MagicMock()
        mock_vector_store_instance.as_retriever.return_value = mock_retriever
        MockNeo4jVector.return_value = mock_vector_store_instance

        handler = QueryHandler(graph=mock_graph, llm=mock_llm, embeddings=mock_embeddings)
        retriever = handler.get_vector_retriever()

        MockNeo4jVector.assert_called_once_with(
            embedding=mock_embeddings,
            graph=mock_graph
        )
        mock_vector_store_instance.as_retriever.assert_called_once()
        assert retriever == mock_retriever

def test_graph_cypher_chain_creation(mock_graph, mock_llm, mock_embeddings):
    """
    Tests that the GraphCypherQAChain is created correctly.
    """
    with patch('src.query_handler.GraphCypherQAChain') as MockCypherChain:
        handler = QueryHandler(graph=mock_graph, llm=mock_llm, embeddings=mock_embeddings)
        chain = handler.get_graph_cypher_chain()

        MockCypherChain.from_llm.assert_called_once_with(
            llm=mock_llm,
            graph=mock_graph,
            verbose=True
        )
        assert chain is not None

def test_full_chain_assembly_and_invocation(mock_graph, mock_llm, mock_embeddings):
    """
    Tests that the full RAG chain is assembled and invokes its components correctly.
    """
    # Arrange
    mock_retriever = MagicMock(spec=Runnable)
    mock_retriever.invoke.return_value = "Vector context"

    mock_cypher_chain = MagicMock(spec=Runnable)
    mock_cypher_chain.invoke.return_value = {"result": "Graph context"}

    # We don't need to patch the prompt template anymore, we'll let the real one be created.
    handler = QueryHandler(graph=mock_graph, llm=mock_llm, embeddings=mock_embeddings)

    # Mock the component getter methods to return our test doubles
    handler.get_vector_retriever = MagicMock(return_value=mock_retriever)
    handler.get_graph_cypher_chain = MagicMock(return_value=mock_cypher_chain)

    # Act
    full_chain = handler.get_full_chain()
    result = full_chain.invoke({"question": "test question"})

    # Assert
    # 1. Check that the component getters were called
    handler.get_vector_retriever.assert_called_once()
    handler.get_graph_cypher_chain.assert_called_once()

    # 2. Check that the retriever and cypher chain were invoked with the question
    mock_retriever.invoke.assert_called_with("test question")
    mock_cypher_chain.invoke.assert_called_with({"query": "test question"})

    # 3. Check that the final LLM was invoked. The prompt content is implicitly tested
    # by the fact that the chain ran without error and called the final LLM.
    mock_llm.invoke.assert_called_once()

    # 4. Check that the final output is the string from the mocked LLM via the StrOutputParser
    assert result == "Final Answer"
