import pytest
from unittest.mock import patch, MagicMock

from src.query_handler import QueryHandler
from langchain_neo4j import Neo4jGraph
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import Runnable
from langchain_core.documents import Document

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
    Tests that the full RAG chain (with GraphCypherQAChain) is assembled
    and invokes its components correctly.
    """
    # Arrange
    # 1. Mock the sub-chains and their return values
    mock_docs = [Document(page_content="This is a test document.")]
    mock_retriever = MagicMock(spec=Runnable)
    mock_retriever.invoke.return_value = mock_docs

    # 2. Mock GraphCypherQAChain
    with patch('src.query_handler.GraphCypherQAChain') as MockGraphCypherQAChain:
        mock_cypher_qa_chain_instance = MagicMock()
        mock_cypher_qa_chain_instance.invoke.return_value = {"query": "test question", "result": "Graph context from Cypher QA Chain"}
        MockGraphCypherQAChain.from_llm.return_value = mock_cypher_qa_chain_instance

        # 3. Set up the handler and mock its component-getter methods
        handler = QueryHandler(graph=mock_graph, llm=mock_llm, embeddings=mock_embeddings)
        handler.get_vector_retriever = MagicMock(return_value=mock_retriever)

        # Act
        full_chain = handler.get_full_chain()
        result = full_chain.invoke({"question": "test question"})

        # Assert
        # 1. Check that the component getters were called
        handler.get_vector_retriever.assert_called_once()

        # 2. Check that the retriever was invoked
        mock_retriever.invoke.assert_called_with("test question")

        # 3. Check that GraphCypherQAChain was initialized and invoked
        MockGraphCypherQAChain.from_llm.assert_called_once_with(
            graph=mock_graph,
            llm=mock_llm,
            verbose=True
        )
        mock_cypher_qa_chain_instance.invoke.assert_called_once_with({"query": "test question"})

        # 4. Check that the final LLM was invoked.
        mock_llm.invoke.assert_called_once()

        # 5. Check that the final output is a dictionary with the correct structure
        assert isinstance(result, dict)
        assert result["answer"] == "Final Answer"  # From the mock_llm fixture
        assert result["graph_data"] == "Graph context from Cypher QA Chain" # From mock_cypher_qa_chain_instance
        assert result["vector_context"] == mock_docs
