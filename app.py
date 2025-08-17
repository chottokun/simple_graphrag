import streamlit as st
import os
import glob
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_neo4j import Neo4jGraph
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from langchain_core.documents import Document
from langchain_ollama.llms import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings
from streamlit_agraph import Node, Edge, agraph, Config

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
    The assistant's message will now contain the text answer and graph data.
    """
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get the RAG chain and invoke it
    rag_chain = st.session_state.query_handler.get_full_chain()
    response = rag_chain.invoke({"question": prompt})  # response is a dict

    # Add assistant response to chat history, including all context
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response.get("answer", "申し訳ありません、回答を生成できませんでした。"),
            "graph_data": response.get("graph_data", []),
            "vector_context": response.get("vector_context", []),
        }
    )

# --- Helper Functions ---
def format_graph_data(graph_data: list[dict]) -> tuple[list[Node], list[Edge]]:
    """
    Converts raw graph data from a Cypher query into lists of Node and Edge
    objects for streamlit-agraph visualization.

    Args:
        graph_data: A list of dictionaries, where each dictionary represents
                    a record from the graph query (e.g., {'n':..., 'r':..., 'm':...}).

    Returns:
        A tuple containing a list of unique Node objects and a list of Edge objects.
    """
    nodes = []
    edges = []
    node_ids = set()

    if not isinstance(graph_data, list):
        return nodes, edges

    for record in graph_data:
        source_node_data = record.get("n")
        target_node_data = record.get("m")
        relationship = record.get("r")

        if not all((source_node_data, target_node_data, relationship)):
            continue

        source_id = source_node_data.get("id")
        target_id = target_node_data.get("id")

        if not all((source_id, target_id)):
            continue

        # Add nodes if they haven't been added yet
        if source_id not in node_ids:
            nodes.append(Node(id=source_id, label=source_id, size=15))
            node_ids.add(source_id)

        if target_id not in node_ids:
            nodes.append(Node(id=target_id, label=target_id, size=15))
            node_ids.add(target_id)

        # Add the edge
        try:
            # Works for real Neo4j relationships and the test mock
            rel_type = relationship.type
        except AttributeError:
            # Fallback for other potential structures
            rel_type = str(relationship)

        edges.append(Edge(source=source_id, target=target_id, label=rel_type))

    return nodes, edges


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
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # If the message is from the assistant, display context sources
            if message["role"] == "assistant":
                graph_context = message.get("graph_data")
                vector_context = message.get("vector_context")

                if graph_context or vector_context:
                    with st.expander("回答の根拠を見る (View Sources)"):
                        tab1, tab2 = st.tabs(
                            ["関連グラフ (Related Graph)", "参照ドキュメント (Referenced Docs)"]
                        )
                        with tab1:
                            if graph_context:
                                nodes, edges = format_graph_data(graph_context)
                                if nodes:
                                    config = Config(
                                        width=750,
                                        height=400,
                                        directed=True,
                                        physics=True,
                                        hierarchical=False,
                                    )
                                    agraph(
                                        nodes=nodes,
                                        edges=edges,
                                        config=config,
                                        key=f"agraph_{i}",
                                    )
                                else:
                                    st.info("関連するグラフデータは見つかりませんでした。")
                            else:
                                st.info("関連するグラフデータは見つかりませんでした。")

                        with tab2:
                            if vector_context:
                                for doc in vector_context:
                                    st.markdown(
                                        f"**Source:** `{doc.metadata.get('source', 'N/A').split('/')[-1]}`"
                                    )
                                    st.markdown(f"> {doc.page_content.replace('_n', ' ')}")
                                    st.divider()
                            else:
                                st.info("参照されたドキュメントはありません。")

    # Accept user input
    if prompt := st.chat_input("質問を入力してください"):
        # Display user message immediately
        st.chat_message("user").markdown(prompt)
        # Handle the query and generate response in the backend
        with st.spinner("考え中..."):
            handle_query(prompt)
        # Rerun the script to display the new assistant message from history
        st.rerun()

if __name__ == "__main__":
    main()