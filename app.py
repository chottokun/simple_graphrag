import streamlit as st
from langchain_neo4j import Neo4jGraph
from langchain_ollama.llms import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings

from src.config import get_neo4j_credentials, get_ollama_config, load_app_config
from src.query_handler import QueryHandler

# It's better to load the config once at the top
load_app_config()

def initialize_session_state():
    """
    Initializes the session state variables if they don't exist.
    Caches the QueryHandler instance.
    """
    if "query_handler" not in st.session_state:
        # Initialize dependencies
        ollama_config = get_ollama_config()
        neo4j_creds = get_neo4j_credentials()

        # In a real app, you'd want error handling here in case services are down
        try:
            graph = Neo4jGraph(
                url=neo4j_creds["uri"],
                username=neo4j_creds["username"],
                password=neo4j_creds["password"],
            )
            llm = OllamaLLM(
                model=ollama_config["model"],
                base_url=ollama_config["base_url"]
            )
            embeddings = OllamaEmbeddings(
                model=ollama_config["embedding_model"],
                base_url=ollama_config["base_url"]
            )
            st.session_state.query_handler = QueryHandler(
                graph=graph, llm=llm, embeddings=embeddings
            )
        except Exception as e:
            st.error(f"Failed to initialize backend services: {e}")
            st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

def handle_query(prompt: str):
    """
    Handles a user query, invokes the RAG chain, and updates the chat history.
    """
    if "query_handler" not in st.session_state:
        st.error("Query handler not initialized. Please refresh.")
        return

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get the RAG chain and invoke it
    rag_chain = st.session_state.query_handler.get_full_chain()
    response = rag_chain.invoke({"question": prompt})

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

def main():
    """The main function that runs the Streamlit application."""
    st.title("グラフRAGシステム - 文書検索チャット")

    # Initialize session state
    initialize_session_state()

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("質問を入力してください"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Handle the query and generate response
        with st.chat_message("assistant"):
            with st.spinner("考え中..."):
                handle_query(prompt)
                # The last message is the new assistant response
                response = st.session_state.messages[-1]["content"]
                st.markdown(response)

# Guard for direct execution
if __name__ == "__main__":
    main()
