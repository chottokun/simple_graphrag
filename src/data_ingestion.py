from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.vectorstores import Neo4jVector
from langchain_neo4j import Neo4jGraph

class DataIngestor:
    """
    Handles the entire data ingestion process.
    This class is designed for dependency injection to facilitate testing.
    """

    def __init__(
        self,
        graph: Neo4jGraph,
        llm: Optional[BaseLanguageModel] = None,
        embeddings: Optional[Embeddings] = None,
    ):
        """
        Initializes the DataIngestor with necessary components.

        Args:
            graph: An instance of Neo4jGraph for DB interaction.
            llm: A language model instance for graph extraction.
            embeddings: An embedding model instance for vector indexing.
        """
        self.graph = graph
        self.llm = llm
        self.embeddings = embeddings
        if llm:
            self.llm_transformer = LLMGraphTransformer(llm=llm)
        else:
            self.llm_transformer = None

    def process_documents(self, documents: List[Document]) -> List:
        """
        Processes a list of documents to extract graph documents.
        Requires the LLM to be set during initialization.
        """
        if not self.llm_transformer:
            raise ValueError("LLM must be provided during initialization to process documents.")
        return self.llm_transformer.convert_to_graph_documents(documents)

    def process_single_document(self, document: Document):
        """
        Processes a single document to extract a graph document.
        Requires the LLM to be set during initialization.
        """
        if not self.llm_transformer:
            raise ValueError("LLM must be provided during initialization to process a document.")
        # The transformer returns a list, so we get the first element
        return self.llm_transformer.convert_to_graph_documents([document])[0]

    def store_graph_documents(self, graph_documents: List) -> None:
        """
        Stores graph documents in the Neo4j database.
        """
        self.graph.add_graph_documents(
            graph_documents,
            baseEntityLabel=True,
            include_source=True
        )

    def create_vector_index(self) -> None:
        """
        Creates a vector index in Neo4j for document chunks.
        Requires the embedding model to be set during initialization.
        """
        if not self.embeddings:
            raise ValueError("Embeddings model must be provided to create a vector index.")

        Neo4jVector.from_existing_graph(
            embedding=self.embeddings,
            graph=self.graph,  # Pass the graph object directly
            search_type="hybrid",
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="embedding",
        )