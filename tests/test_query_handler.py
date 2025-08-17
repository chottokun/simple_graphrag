import pytest
from unittest.mock import patch, MagicMock

from src.query_handler import QueryHandler
from langchain_neo4j import Neo4jGraph
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import Runnable

from langchain_core.messages import AIMessage


@pytest.fixture
def mock_llm():
    """Fixture for a mocked LLM."""
    llm = MagicMock(spec=BaseLanguageModel)
    # A real LLM returns a message object, which the StrOutputParser handles.
    llm.invoke.return_value = AIMessage(content="Final Answer")
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

def test_full_chain_assembly_and_invocation(mock_graph, mock_llm, mock_embeddings):
    """
    Tests that the full RAG chain is assembled and invokes its components correctly,
    returning a dictionary with an answer and graph data.
    """
    # Arrange
    # 1. Mock the sub-chains and their return values
    mock_retriever = MagicMock(spec=Runnable)
    mock_retriever.invoke.return_value = "Vector context"

    # The cypher generation chain should return a string (the cypher query)
    mock_cypher_generation_chain = MagicMock(spec=Runnable)
    mock_cypher_generation_chain.invoke.return_value = "MATCH (n) RETURN n"

    # 2. Mock the graph query result
    mock_graph_data = [{"n": "a", "r": "b", "m": "c"}]
    mock_graph.query.return_value = mock_graph_data

    # 3. Set up the handler and mock its component-getter methods
    handler = QueryHandler(graph=mock_graph, llm=mock_llm, embeddings=mock_embeddings)
    handler.get_vector_retriever = MagicMock(return_value=mock_retriever)
    handler.get_cypher_generation_chain = MagicMock(
        return_value=mock_cypher_generation_chain
    )

    # Act
    full_chain = handler.get_full_chain()
    result = full_chain.invoke({"question": "test question"})

    # Assert
    # 1. Check that the component getters were called
    handler.get_vector_retriever.assert_called_once()
    handler.get_cypher_generation_chain.assert_called_once()

    # 2. Check that the retriever and cypher chain were invoked
    mock_retriever.invoke.assert_called_with("test question")
    mock_cypher_generation_chain.invoke.assert_called_with({"question": "test question"})

    # 3. Check that the graph query was run with the generated cypher
    mock_graph.query.assert_called_once_with("MATCH (n) RETURN n")

    # 4. Check that the final LLM was invoked.
    mock_llm.invoke.assert_called_once()

    # 5. Check that the final output is a dictionary with the correct structure
    assert isinstance(result, dict)
    assert "answer" in result
    assert "graph_data" in result
    assert result["answer"] == "Final Answer"  # From the mock_llm fixture
    assert result["graph_data"] == mock_graph_data
