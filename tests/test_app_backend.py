import pytest
from unittest.mock import patch, MagicMock
import streamlit as st

# Import the functions to be tested from app.py
from app import handle_query, initialize_session_state

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
    mock_chain.invoke.return_value = "This is the final answer."

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

@patch('app.Neo4jGraph')
@patch('app.OllamaLLM')
@patch('app.OllamaEmbeddings')
@patch('app.QueryHandler')
def test_initialize_session_state(MockQueryHandler, MockEmbeddings, MockLLM, MockNeo4jGraph):
    """
    Tests that the session state is initialized correctly by mocking all external dependencies.
    """
    # Act
    initialize_session_state()

    # Assert
    # Check that dependencies were instantiated
    MockNeo4jGraph.assert_called_once()
    MockLLM.assert_called_once()
    MockEmbeddings.assert_called_once()

    # Check that QueryHandler was instantiated with the mocked dependencies
    MockQueryHandler.assert_called_once_with(
        graph=MockNeo4jGraph.return_value,
        llm=MockLLM.return_value,
        embeddings=MockEmbeddings.return_value
    )

    # Check that session state was set
    assert "messages" in st.session_state
    assert st.session_state.messages == []
    assert "query_handler" in st.session_state
    assert st.session_state.query_handler == MockQueryHandler.return_value
