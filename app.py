import streamlit as st
import os
import glob
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_neo4j import Neo4jGraph
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from langchain_core.documents import Document
from langchain_ollama.llms import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings

from src.config import get_neo4j_credentials, get_ollama_config, load_app_config
from src.query_handler import QueryHandler
from src.data_ingestion import DataIngestor

# Load configuration at the top
load_app_config()

# --- Caching Resource Initialization ---
@st.cache_resource
def get_neo4j_graph():
    """Initializes and returns a Neo4jGraph instance."""
    neo4j_creds = get_neo4j_credentials()
    try:
        return Neo4jGraph(
            url=neo4j_creds["uri"],
            username=neo4j_creds["username"],
            password=neo4j_creds["password"],
        )
    except Exception as e:
        st.error(f"Failed to connect to Neo4j: {e}")
        st.stop()

@st.cache_resource
def get_llm():
    """Initializes and returns an OllamaLLM instance."""
    ollama_config = get_ollama_config()
    try:
        return OllamaLLM(
            model=ollama_config["model"], base_url=ollama_config["base_url"]
        )
    except Exception as e:
        st.error(f"Failed to connect to Ollama (LLM): {e}")
        st.stop()

@st.cache_resource
def get_embeddings():
    """Initializes and returns an OllamaEmbeddings instance."""
    ollama_config = get_ollama_config()
    try:
        return OllamaEmbeddings(
            model=ollama_config["embedding_model"],
            base_url=ollama_config["base_url"],
        )
    except Exception as e:
        st.error(f"Failed to connect to Ollama (Embeddings): {e}")
        st.stop()

# --- Session State Initialization ---
def initialize_session_state():
    """Initializes session state variables."""
    if "query_handler" not in st.session_state:
        st.session_state.query_handler = QueryHandler(
            graph=get_neo4j_graph(), llm=get_llm(), embeddings=get_embeddings()
        )
    if "messages" not in st.session_state:
        st.session_state.messages = []

# --- Data Ingestion Logic ---
def handle_ingestion():
    """
    Handles the data ingestion process based on files in the ./data directory.
    Provides detailed progress feedback to the Streamlit UI.
    """
    data_path = "./data"
    if not os.path.exists(data_path):
        st.sidebar.warning(f"'{data_path}' directory not found.")
        return

    md_files = glob.glob(os.path.join(data_path, "*.md"))
    pdf_files = glob.glob(os.path.join(data_path, "*.pdf"))

    if not md_files and not pdf_files:
        st.sidebar.warning(f"No Markdown or PDF files found in '{data_path}'.")
        return

    with st.status("データ投入を開始します...", expanded=True) as status:
        try:
            # --- ステップ0: ドキュメントの読み込み ---
            status.update(label="ステップ 0/4: ドキュメントを読み込んでいます...")
            documents = []
            for md_file in md_files:
                loader = UnstructuredMarkdownLoader(md_file)
                documents.extend(loader.load())
            for pdf_file in pdf_files:
                loader = PyPDFLoader(pdf_file)
                documents.extend(loader.load())

            if not documents:
                status.update(label="ドキュメントの読み込みに失敗しました。", state="error", expanded=False)
                st.sidebar.error("Failed to load any documents.")
                return
            st.write(f"✓ {len(documents)}個のドキュメントを読み込みました。")

            # --- ドキュメントの分割 ---
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
            split_documents = text_splitter.split_documents(documents)
            st.write(f"✓ {len(documents)}個のドキュメントを{len(split_documents)}個のチャンクに分割しました。")

            ingestor = DataIngestor(
                graph=get_neo4j_graph(), llm=get_llm(), embeddings=get_embeddings()
            )

            # --- ステップ1: グラフ構造の処理 (最も時間がかかる部分) ---
            status.update(label=f"ステップ 1/4: {len(split_documents)}個のチャンクをグラフ構造に変換しています...")
            
            graph_documents = []
            progress_bar = st.progress(0, text="チャンクの処理状況")
            
            for i, doc in enumerate(split_documents):
                graph_doc = ingestor.process_single_document(doc)
                graph_documents.append(graph_doc)
                progress = (i + 1) / len(split_documents)
                progress_bar.progress(progress, text=f"チャンク {i + 1}/{len(split_documents)} を処理中")

            st.write(f"✓ {len(split_documents)}個のチャンクから{len(graph_documents)}個のグラフドキュメントを生成しました。")

            # --- ステップ2: Neo4jへの格納 ---
            status.update(label="ステップ 2/4: グラフドキュメントをNeo4jに格納しています...")
            ingestor.store_graph_documents(graph_documents)
            st.write("✓ グラフドキュメントをNeo4jに正常に格納しました。")

            # --- ステップ3: ベクトルインデックスの作成 ---
            status.update(label="ステップ 3/4: ベクトルインデックスを作成しています... (時間がかかる場合があります)")
            ingestor.create_vector_index()
            st.write("✓ ベクトルインデックスを正常に作成しました。")

            # --- 完了 ---
            status.update(label="データ投入が完了しました！", state="complete", expanded=False)
            st.sidebar.success(f"Successfully ingested {len(documents)} documents!")

        except Exception as e:
            status.update(label=f"エラーが発生しました: {e}", state="error", expanded=True)
            st.sidebar.error(f"An error occurred during ingestion: {e}")

# --- Chat Logic ---
def handle_query(prompt: str):
    """
    Handles a user query, invokes the RAG chain, and updates the chat history.
    """
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get the RAG chain and invoke it
    rag_chain = st.session_state.query_handler.get_full_chain()
    response = rag_chain.invoke({"question": prompt})

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# --- Main Application ---
def main():
    """The main function that runs the Streamlit application."""
    st.title("graph RAG. chat.")

    # --- Sidebar for Data Ingestion ---
    with st.sidebar:
        st.header("データ投入")
        st.markdown(
            "ローカルの`./data`ディレクトリにあるMarkdownやPDFファイルをNeo4jに投入し、ベクトルインデックスを作成します。"
        )
        if st.button("データを取り込む"):
            handle_ingestion()

    # Initialize session state
    initialize_session_state()

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("質問を入力してください"):
        st.chat_message("user").markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("考え中..."):
                handle_query(prompt)
                response = st.session_state.messages[-1]["content"]
                st.markdown(response)

if __name__ == "__main__":
    main()