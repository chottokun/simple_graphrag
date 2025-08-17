# Hybrid Graph RAG System (Neo4j, LangChain, Ollama)

This project implements an advanced Retrieval-Augmented Generation (RAG) chat system using a hybrid approach that combines vector search and knowledge graph search. It leverages Neo4j for graph-based data storage and retrieval, LangChain for orchestrating LLM interactions, and Ollama for running local language models. The frontend is built with Streamlit, allowing users to ingest documents and interact with the chat system through a user-friendly interface.

## Project Goal

The primary goal of this project is to build a sophisticated RAG system that goes beyond simple keyword or vector similarity search. By constructing a knowledge graph from unstructured documents, the system can understand and query the relationships between entities, providing more contextually aware and accurate answers. This hybrid model aims to answer complex questions that require understanding the underlying connections within the source data.

## Key Features

-   **Document Ingestion**: Upload Markdown (`.md`) or PDF (`.pdf`) files directly through the Streamlit UI.
-   **Automated Knowledge Graph Construction**: Documents are processed by an LLM to automatically extract entities and relationships, which are then stored in a Neo4j graph.
-   **Hybrid Search**: User queries trigger both a vector search for semantic similarity and a graph search for contextual relationships.
-   **Interactive Q&A**: An LLM generates answers based on the combined context from both search methods.
-   **Source Visualization**: The UI displays the retrieved knowledge graph and the source document chunks that were used to generate the answer, ensuring transparency and traceability.

## System Architecture

The system is composed of two main workflows: Data Ingestion and Query Handling.

### 1. Data Ingestion Flow

1.  **File Upload**: User uploads documents via the Streamlit sidebar.
2.  **Document Loading & Splitting**: The system loads the content and splits it into smaller, manageable chunks.
3.  **Graph Extraction**: Each chunk is passed to an LLM (`LLMGraphTransformer`) which extracts entities (nodes) and relationships (edges).
4.  **Graph Storage**: The extracted graph data, along with the original text chunk (as a `Document` node), is stored in the Neo4j database.
5.  **Vector Indexing**: The text content of each `Document` node is converted into a vector embedding and stored in a Neo4j vector index for efficient similarity search.

### 2. Query Handling Flow

1.  **User Query**: A user asks a question through the Streamlit chat interface.
2.  **Entity Extraction**: The user's query is sent to an LLM to extract key entities.
3.  **Hybrid Retrieval**:
    *   **Vector Search**: The entire query is used to find the most semantically similar `Document` nodes from the vector index.
    *   **Graph Search**: The extracted entities are used to query the Neo4j graph, retrieving related nodes and relationships.
4.  **Context Consolidation**: The results from both vector and graph searches are combined into a comprehensive context.
5.  **Answer Generation**: The consolidated context and the original query are passed to an LLM, which generates a final, context-aware answer.
6.  **Display**: The answer, along with the visual graph context and source text chunks, is displayed in the UI.

## Project Structure

```
/
├── app.py                    # Main Streamlit application file
├── src/                      # Core application logic
│   ├── config.py             # Environment variable and configuration management
│   ├── data_ingestion.py     # Handles the data ingestion pipeline
│   └── query_handler.py      # Handles the query processing and RAG chain
├── tests/                    # Pytest unit and integration tests
│   ├── test_app_backend.py   # Tests for Streamlit backend logic
│   ├── test_connections.py   # Online tests for external service connections
│   ├── test_data_ingestion.py# Tests for the data ingestion module
│   └── test_query_handler.py # Tests for the query handler module
├── data/                     # (Git-ignored) Directory for source documents
├── docs/                     # Project documentation
├── .env.example              # Example environment variables file
├── requirements.txt          # Python dependencies
└── neo4j-docker-compose.yaml # Docker Compose for Neo4j & Ollama
```

## Setup and Execution

### 1. Prerequisites

-   Docker and Docker Compose
-   Python 3.9+
-   Access to an Ollama-compatible LLM

### 2. Environment Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd simple_graphrag
    ```

2.  **Set up environment variables:**
    Copy the example `.env` file and customize it with your settings.
    ```bash
    cp .env.example .env
    ```
    The default values in `.env.example` are configured to work with the provided `neo4j-docker-compose.yaml`. You will need to ensure the Ollama models are available at your `OLLAMA_BASE_URL`.

    **Required `.env` variables:**
    -   `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`
    -   `OLLAMA_BASE_URL`: The base URL for your Ollama instance.
    -   `OLLAMA_MODEL`: The name of the main chat model (e.g., `gemma:2b-instruct-q4_0`).
    -   `OLLAMA_EMBEDDING_MODEL`: The name of the embedding model (e.g., `mxbai-embed-large`).

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### 3. Running the Application

1.  **Start external services (Neo4j & Ollama):**
    Use the provided Docker Compose file to start the necessary services.
    ```bash
    docker-compose -f neo4j-docker-compose.yaml up -d
    ```

2.  **Pull Ollama models:**
    Make sure the models specified in your `.env` file are available in your Ollama instance.
    ```bash
    ollama pull gemma:2b-instruct-q4_0
    ollama pull mxbai-embed-large
    ```

3.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
    The application will be available at `http://localhost:8501`.

### 4. Using the Application

1.  **Create a `data` directory** in the project root.
2.  Place the Markdown or PDF files you want to process inside the `data` directory.
3.  Open the Streamlit application in your browser.
4.  Click the **"データを取り込む" (Ingest Data)** button in the sidebar.
5.  Once ingestion is complete, you can start asking questions about your documents in the chat window.

## Testing

This project uses `pytest` for testing. The tests are divided into unit tests (which use mocks) and online tests (which require live connections to services).

-   **Run all unit tests:**
    ```bash
    pytest
    ```

-   **Run online connection tests:**
    These tests verify the connection to live Neo4j and Ollama instances. Make sure the services are running before executing.
    ```bash
    pytest -m online
    ```
