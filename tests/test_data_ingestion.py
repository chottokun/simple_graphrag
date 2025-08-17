import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_neo4j import Neo4jGraph

from src.data_ingestion import DataIngestor

@pytest.fixture
def mock_graph():
    """Fixture for a mocked Neo4jGraph instance."""
    return MagicMock(spec=Neo4jGraph)

@pytest.fixture
def mock_llm():
    """Fixture for a mocked LLM."""
    return MagicMock()

@pytest.fixture
def mock_embeddings():
    """Fixture for a mocked Embeddings model."""
    return MagicMock()

@pytest.fixture
def sample_document():
    """A sample LangChain Document to be processed."""
    return Document(page_content="LangChain makes LLM application development easy.")

@pytest.fixture
def sample_graph_document():
    """A sample GraphDocument to be stored."""
    nodes = [Node(id="LangChain", type="Entity"), Node(id="LLM application development", type="Task")]
    relationships = [Relationship(source=nodes[0], target=nodes[1], type="MAKES_EASY")]
    source_doc = Document(page_content="LangChain makes LLM application development easy.")
    return GraphDocument(nodes=nodes, relationships=relationships, source=source_doc)

def test_data_ingestor_initialization(mock_graph):
    """Test that the DataIngestor class initializes correctly."""
    ingestor = DataIngestor(graph=mock_graph)
    assert ingestor.graph == mock_graph

def test_process_documents(mock_graph, mock_llm, sample_document, sample_graph_document):
    """
    Tests if a single document is correctly converted into a GraphDocument.
    """
    with patch('src.data_ingestion.LLMGraphTransformer') as MockTransformer:
        mock_transformer_instance = MagicMock()
        mock_transformer_instance.convert_to_graph_documents.return_value = [sample_graph_document]
        MockTransformer.return_value = mock_transformer_instance

        ingestor = DataIngestor(graph=mock_graph, llm=mock_llm)
        graph_documents = ingestor.process_documents([sample_document])

        MockTransformer.assert_called_once_with(llm=mock_llm)
        assert graph_documents[0] == sample_graph_document
        mock_transformer_instance.convert_to_graph_documents.assert_called_once_with([sample_document])

def test_store_graph_documents(mock_graph, sample_graph_document):
    """
    Tests that the store_graph_documents method calls add_graph_documents.
    """
    ingestor = DataIngestor(graph=mock_graph)
    ingestor.store_graph_documents([sample_graph_document])

    mock_graph.add_graph_documents.assert_called_once_with(
        [sample_graph_document],
        baseEntityLabel=True,
        include_source=True
    )

def test_create_vector_index(mock_graph, mock_embeddings):
    """
    Tests the create_vector_index method.
    """
    with patch('src.data_ingestion.Neo4jVector') as MockNeo4jVector:
        ingestor = DataIngestor(graph=mock_graph, embeddings=mock_embeddings)
        ingestor.create_vector_index()

        MockNeo4jVector.from_existing_graph.assert_called_once_with(
            embedding=mock_embeddings,
            graph=mock_graph,
            search_type="hybrid",
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="embedding",
        )
