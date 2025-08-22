import pytest
from unittest.mock import patch, MagicMock
import streamlit as st
from streamlit_agraph import Node, Edge
from langchain_core.documents import Document

# Import the functions to be tested from app.py
from app import handle_query, initialize_session_state, format_graph_data

@pytest.fixture(autouse=True)
def clean_session_state():
    """Fixture to ensure a clean session state for each test."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    yield

def test_handle_query():
    """
    Tests the handle_query function to ensure it interacts with QueryHandler
    and updates session_state correctly.
    """
    # Arrange
    mock_chain = MagicMock()
    # The chain now returns a dictionary with all context
    mock_docs = [Document(page_content="doc content")]
    mock_chain.invoke.return_value = {
        "answer": "This is the final answer.",
        "graph_data": [{"n": "a", "r": "b", "m": "c"}],
        "vector_context": mock_docs,
    }

    mock_query_handler = MagicMock()
    mock_query_handler.get_full_chain.return_value = mock_chain

    # Manually set up the session state as it would be in the running app
    st.session_state.query_handler = mock_query_handler
    st.session_state.messages = []

    test_prompt = "What is GraphRAG?"

    # Act
    handle_query(test_prompt)

    # Assert
    # 1. The full chain was retrieved and invoked
    mock_query_handler.get_full_chain.assert_called_once()
    mock_chain.invoke.assert_called_once_with({"question": test_prompt})

    # 2. The user message and assistant response were added to session_state
    assert len(st.session_state.messages) == 2
    assert st.session_state.messages[0]["role"] == "user"
    assert st.session_state.messages[0]["content"] == test_prompt
    assert st.session_state.messages[1]["role"] == "assistant"
    assert st.session_state.messages[1]["content"] == "This is the final answer."
    assert "graph_data" in st.session_state.messages[1]
    assert st.session_state.messages[1]["graph_data"] is not None
    assert "vector_context" in st.session_state.messages[1]
    assert st.session_state.messages[1]["vector_context"] == mock_docs

@patch('app.Neo4jGraph')
@patch('app.get_llm_and_embeddings_cached')
@patch('app.QueryHandler')
def test_initialize_session_state(MockQueryHandler, mock_get_llm_and_embeddings_cached, MockNeo4jGraph):
    """
    Tests that the session state is initialized correctly by mocking all external dependencies.
    """
    # Arrange
    mock_llm = MagicMock()
    mock_embeddings = MagicMock()
    mock_get_llm_and_embeddings_cached.return_value = (mock_llm, mock_embeddings)

    # Act
    initialize_session_state()

    # Assert
    # Check that dependencies were instantiated
    MockNeo4jGraph.assert_called_once()
    mock_get_llm_and_embeddings_cached.assert_called_once()

    # Check that QueryHandler was instantiated with the mocked dependencies
    MockQueryHandler.assert_called_once_with(
        graph=MockNeo4jGraph.return_value,
        llm=mock_llm,
        embeddings=mock_embeddings
    )

    # Check that session state was set
    assert "messages" in st.session_state
    assert st.session_state.messages == []
    assert "query_handler" in st.session_state
    assert st.session_state.query_handler == MockQueryHandler.return_value


def test_graph_data_formatting():
    """
    Tests the formatting of raw Cypher results into lists of
    agraph Nodes and Edges. It should handle empty lists and avoid duplicates.
    """
    # Arrange: Mock a raw Cypher query result.
    # The value for a relationship needs to be mocked.
    class MockRelationship:
        def __init__(self, type):
            self.type = type

    mock_cypher_result = [
        {
            "n": {"id": "LangChain"},
            "r": MockRelationship("IS_A"),
            "m": {"id": "Framework"},
        },
        {
            "n": {"id": "LangChain"},
            "r": MockRelationship("USES"),
            "m": {"id": "LLM"},
        },
        # This record introduces a duplicate node ("LangChain")
        {
            "n": {"id": "LLM"},
            "r": MockRelationship("IS_USED_BY"),
            "m": {"id": "LangChain"},
        },
    ]

    # Act
    nodes, edges = format_graph_data(mock_cypher_result)

    # Assert
    # 1. Check for correct types
    assert all(isinstance(n, Node) for n in nodes)
    assert all(isinstance(e, Edge) for e in edges)

    # 2. Check for correct number of unique nodes and edges
    # Unique nodes: "LangChain", "Framework", "LLM" -> 3
    assert len(nodes) == 3
    # Edges: 3 records should produce 3 edges
    assert len(edges) == 3

    # 3. Check node IDs for uniqueness and correctness
    node_ids = {n.id for n in nodes}
    expected_node_ids = {"LangChain", "Framework", "LLM"}
    assert node_ids == expected_node_ids

    # 4. Check edge properties
    edge_tuples = {(e.source, e.label, e.to) for e in edges}
    assert ("LangChain", "IS_A", "Framework") in edge_tuples
    assert ("LangChain", "USES", "LLM") in edge_tuples
    assert ("LLM", "IS_USED_BY", "LangChain") in edge_tuples

def test_graph_data_formatting_empty_input():
    """
    Tests that the formatting function handles an empty list gracefully.
    """
    # Arrange
    empty_result = []

    # Act
    nodes, edges = format_graph_data(empty_result)

    # Assert
    assert nodes == []
    assert edges == []
