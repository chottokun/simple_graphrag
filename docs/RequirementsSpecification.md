### **Neo4jとLangChainを用いた高度なRAGシステム設計書**

#### **1. はじめに**

本ドキュメントは、社内文書や学術論文といった非構造化テキストデータを対象とし、高度な情報探索を可能にするチャット型システムの設計を定義するものです。本システムは、ベクトル検索の類似性マッチングと、知識グラフが持つ文脈的な関係性の両方を活用する**グラフRAG（Graph-Augmented Generation）**アプローチを採用します。これにより、単なるキーワード検索を超え、データ間の隠れた関係性を捉えた深い洞察を提供することを目指します。

技術スタックの中核として、グラフデータベースの**Neo4j**と、LLMアプリケーション開発フレームワークの**LangChain**を採用し、フロントエンドには**Streamlit**を利用します。

#### **2. 要件定義**

ご提示いただいた要件定義は明確であり、これを本システムの正式な要件とします。

*   **ユースケース／業務領域**: 社内文書検索および学術論文検索のチャット型情報探索システム。
*   **データソース**: PDF文書、Markdownファイル。
*   **機能要件**:
    1.  自然言語クエリに基づく関連情報（文書チャンクとグラフ情報）の検索。
    2.  検索結果を基にしたLLMによる回答生成。
    3.  対話履歴を考慮したインタラクティブな応答。
*   **UI要件 (Streamlit)**:
    1.  チャット形式の検索入力欄。
    2.  回答の根拠として、関連知識グラフのインタラクティブな可視化と、参照されたソースドキュメント（テキストチャンク）の一覧をタブ形式で表示する。
    3.  LLMが生成した回答と対話履歴の表示領域。
*   **非機能要件**: 
    *   **プロトタイプフェーズ**: 応答速度や同時接続数に厳しい制約は設けない。
    *   **将来的な目標**:
        *   **応答速度**: 通常のクエリに対して、90%が5秒以内に応答を完了する。
        *   **同時接続数**: まずは10人程度の同時利用を想定し、安定稼働を目指す。

#### **3. システムアーキテクチャ設計**

本システムは、「データインジェスト（事前処理）」と「検索・応答（リアルタイム処理）」の2つの主要なフェーズで構成されます。

**3.1. 技術スタック**

*   **グラフデータベース & ベクトルストア**: Neo4j (Community or AuraDB)
*   **LLMアプリケーションフレームワーク**: LangChain
*   **大規模言語モデル (LLM)**: 設定によりOllama, OpenAI, Azure OpenAIなどを利用可能。
*   **埋め込みモデル**: 設定によりOllama, OpenAIなどを利用可能。
*   **フロントエンド**: Streamlit
*   **グラフ可視化**: Pyvis, streamlit-agraph

本設計では、環境変数 `LLM_PROVIDER` を設定することで、利用するLLMおよび埋め込みモデルを切り替えることができます。

**3.2. 処理フロー概要**

1.  **データインジェスト (バッチ処理)**
    *   **① データ収集・前処理**: PDFやMarkdown文書を読み込み、意味のある単位（チャンク）に分割します。
    *   **② ナレッジグラフ構築**: 各チャンクからLLMを用いてエンティティと関係性を抽出し、Neo4jに知識グラフとして格納します。この際、元のチャンクも`Document`ノードとしてグラフに統合します。抽出時には、`LLMGraphTransformer`の`allowed_nodes`および`allowed_relationships`パラメータを用いて、抽出対象を制御し、精度と一貫性を向上させます。
    *   **③ 埋め込みとインデックス作成**: `Document`ノードの内容をベクトル化し、Neo4j上にベクトル検索インデックスとキーワード検索インデックスを構築します。

2.  **検索・応答 (リアルタイム処理)**
    *   **④ クエリ入力**: ユーザーがStreamlit UIから自然言語で質問します。
    *   **⑤ ハイブリッド検索**: クエリを解析し、「ベクトル検索」と「グラフ検索」を並行して実行し、関連情報を収集します。
    *   **⑥ コンテキスト統合**: 収集した情報を統合し、LLMへの入力（プロンプト）を生成します。
    *   **⑦ 回答生成と表示**: LLMが生成した回答と、関連グラフの可視化をUIに表示します。

---

#### **4. 実装フェーズ詳細とサンプルコード**

以下に、主要な実装フェーズごとの詳細な設計と、具体的なサンプルコードを示します。

**4.1. フェーズ1: データインジェストと知識グラフ構築**

非構造化テキストから、分析可能な構造化データ（知識グラフ）を半自動で構築する、本システムの根幹となるフェーズです。

*   **手順**:
    1.  `UnstructuredFileLoader`等で多様な形式の文書を読み込みます。
    2.  `RecursiveCharacterTextSplitter`で適切なサイズのチャンクに分割します。
    3.  `LLMGraphTransformer`を使い、各チャンクからノードと関係性を抽出します。この際、環境変数`GRAPH_ALLOWED_NODES`と`GRAPH_ALLOWED_RELATIONSHIPS`で指定されたノードタイプと関係性タイプのみを抽出対象とすることで、抽出の精度と一貫性を高めます。
    4.  `graph.add_graph_documents`でNeo4jに格納します。`include_source=True`で引用元情報が、`baseEntityLabel=True`で全エンティティへの共通ラベルが付与され、後の処理が容易になります。

*   **サンプルコード1: グラフ構築**
    ```python
    import os
    from langchain_neo4j import Neo4jGraph
    from langchain_experimental.graph_transformers import LLMGraphTransformer
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from src.llm_factory import get_llm_and_embeddings
    from src.config import get_graph_transformer_config

    # --- 1. Neo4jへの接続設定 (環境変数から読み込み) ---
    # os.environ["NEO4J_URI"] = "bolt://localhost:7687"
    # os.environ["NEO4J_USERNAME"] = "neo4j"
    # os.environ["NEO4J_PASSWORD"] = "your_password"
    # os.environ["LLM_PROVIDER"] = "ollama" # または "openai", "azure_openai"
    # os.environ["OLLAMA_MODEL"] = "gemma2"
    # os.environ["OLLAMA_EMBEDDING_MODEL"] = "nomic-embed-text"
    # os.environ["GRAPH_ALLOWED_NODES"] = "Person,Organization,Location,Event,Concept,Document"
    # os.environ["GRAPH_ALLOWED_RELATIONSHIPS"] = "HAS_RELATIONSHIP,LOCATED_IN,PART_OF,MENTIONS,RELATES_TO"

    graph = Neo4jGraph()
    llm, _ = get_llm_and_embeddings() # LLMのみ取得

    # --- 2. ドキュメントの読み込みと分割 ---
    loader = PyPDFLoader("path/to/your/document.pdf")
    documents = loader.load_and_split(
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    )

    # --- 3. グラフ抽出と格納 ---
    graph_transformer_config = get_graph_transformer_config()
    llm_transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=graph_transformer_config.get("allowed_nodes"),
        allowed_relationships=graph_transformer_config.get("allowed_relationships")
    )

    # ドキュメントからグラフ表現を抽出
    graph_documents = llm_transformer.convert_to_graph_documents(documents)

    # Neo4jにグラフを格納
    graph.add_graph_documents(
        graph_documents,
        baseEntityLabel=True, # 全ノードに "__Entity__" ラベルを追加
        include_source=True   # 元のDocumentノードと :MENTIONS 関係で接続
    )
    print("知識グラフの構築が完了しました。")
    ```

**4.2. フェーズ2: 埋め込みとハイブリッド検索インデックスの構築**

グラフ構造に加えて、テキストのセマンティックな意味を捉えるためのベクトル検索インフラを整備します。

*   **手順**:
    1.  `Neo4jVector.from_existing_graph`メソッドを利用します。このメソッドは非常に強力で、以下の処理を自動で行います。
        *   指定したノード（ここでは`:Document`）に対するベクトルインデックスを作成。
        *   指定したプロパティ（`text`）の内容を埋め込みモデルでベクトル化。
        *   キーワード検索用の全文検索インデックスも同時に作成。

*   **サンプルコード2: インデックス構築**
    ```python
    from langchain_community.vectorstores import Neo4jVector
    from src.llm_factory import get_llm_and_embeddings
    from langchain_neo4j import Neo4jGraph

    # (上記で初期化した graph を利用)
    _, embeddings = get_llm_and_embeddings() # 埋め込みモデルのみ取得
    graph = Neo4jGraph() # Neo4jGraphのインスタンス化

    # --- ベクトルストアとインデックスの作成 ---
    vector_store = Neo4jVector.from_existing_graph(
        embedding=embeddings,
        graph=graph,
        search_type="hybrid", # キーワード検索とベクトル検索の両方を利用
        node_label="Document", # 対象となるノードのラベル
        text_node_properties=["text"], # テキスト内容が格納されているプロパティ
        embedding_node_property="embedding", # ベクトルを格納するプロパティ名
    )
    print("ベクトルインデックスとキーワードインデックスの構築が完了しました。")
    ```

**4.3. フェーズ3: 検索・応答パイプライン**

ユーザーの質問に対して、構築したグラフとインデックスをフル活用して回答を生成する、システムの心臓部です。

*   **手順**:
    1.  **ハイブリッド検索**: ユーザーの質問から、`GraphCypherQAChain`を用いて関連するグラフ情報を抽出し、同時にベクトル検索で関連ドキュメントチャンクを取得します。
    2.  **コンテキスト統合と回答生成**: 収集した情報を統合し、LLMへの入力（プロンプト）を生成します。最終的な回答はLLMに生成させます。

*   **サンプルコード3: 検索と応答生成チェーン（GraphCypherQAChain方式）**
    ```python
    from operator import itemgetter
    from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
    from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda
    from langchain_core.output_parsers import StrOutputParser
    from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
    from src.llm_factory import get_llm_and_embeddings
    from langchain_neo4j import Neo4jGraph, Neo4jVector

    # (上記で初期化した graph, llm, embeddings を利用)
    llm, embeddings = get_llm_and_embeddings()
    graph = Neo4jGraph()

    # --- 1. ベクトル検索用のリトリーバー ---
    vector_store = Neo4jVector(
        embedding=embeddings,
        graph=graph
    )
    vector_retriever = vector_store.as_retriever()

    # --- 2. GraphCypherQAChainの初期化 ---
    cypher_qa_chain = GraphCypherQAChain.from_llm(
        graph=graph,
        llm=llm,
        verbose=True # デバッグ用にTrueに設定
    )

    # --- 3. 全体を統合したチェーン ---
    def retrieve_all_data(inputs: dict) -> dict:
        question = inputs["question"]
        
        vector_context = vector_retriever.invoke(question)
        
        # GraphCypherQAChainでグラフ情報を取得
        try:
            cypher_qa_result = cypher_qa_chain.invoke({"query": question})
            graph_context = cypher_qa_result.get("result", "")
            # 可視化用に生のグラフデータを取得する必要がある場合、
            # cypher_qa_chainの内部で生成されたCypherクエリを解析するか、
            # 別の方法でグラフデータを取得するロジックを追加検討。
            # ここでは簡略化のため、cypher_qa_resultをそのまま利用。
            graph_data_for_viz = graph_context # 仮のデータ、要改善
        except Exception as e:
            print(f"GraphCypherQAChain failed: {e}")
            graph_context = ""
            graph_data_for_viz = []

        return {
            "question": question,
            "vector_context": vector_context,
            "graph_context": graph_context,
            "graph_data_for_viz": graph_data_for_viz,
        }

    prompt_template = PromptTemplate.from_template(
        """
あなたは社内文書や学術論文に詳しいアシスタントです。
以下のコンテキスト情報を利用して、質問に答えてください。
**ベクトル検索の結果 (文書の断片):**
{vector_context}
**グラフ検索の結果 (関連エンティティや関係):**
{graph_context}
**質問:** {question}
**回答:**
"""
    )
    
    answer_chain = (
        itemgetter("context")
        | prompt_template
        | llm
        | StrOutputParser()
    )

    rag_chain = (
        RunnablePassthrough.assign(context=RunnableLambda(retrieve_all_data))
        .assign(answer=answer_chain)
        | RunnableLambda(
            lambda x: {
                "answer": x["answer"],
                "graph_data": x["context"]["graph_data_for_viz"],
                "vector_context": x["context"]["vector_context"],
            }
        )
    )

    # --- 4. チェーンの実行 ---
    question = "LangChainとNeo4jの関係について教えてください。"
    response = rag_chain.invoke({"question": question})
    print(f"Answer: {response['answer']}")
    print(f"Graph Data: {response['graph_data']}")
    ```

**4.4. フェーズ4: UI (Streamlit) の実装**

ユーザーとの対話を実現するフロントエンドを構築します。

*   **手順**:
    1.  `st.chat_input`と`st.chat_message`を使い、基本的なチャットUIを実装します。
    2.  セッション状態（`st.session_state`）を利用して、会話履歴を保持します。
    3.  バックエンドのRAGチェーンを呼び出し、回答と関連コンテキスト（グラフデータ、ベクトル検索結果）を取得します。
    4.  回答のテキストを表示し、`st.expander`と`st.tabs`を用いて、回答の根拠となった「関連グラフ」と「参照ドキュメント」を切り替えて表示できるようにします。

*   **サンプルコード4: Streamlit UIの骨格**
    ```python
    import streamlit as st
    from streamlit_agraph import agraph, Node, Edge, Config
    from src.llm_factory import get_llm_and_embeddings
    from src.query_handler import QueryHandler
    from src.data_ingestion import DataIngestor
    from src.config import get_neo4j_credentials, load_app_config
    from langchain_neo4j import Neo4jGraph
    import os
    import glob
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader

    # アプリケーション設定のロード
    load_app_config()

    # --- キャッシュされたリソースの初期化 ---
    @st.cache_resource
    def get_neo4j_graph():
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
    def get_llm_and_embeddings_cached():
        return get_llm_and_embeddings()

    # --- セッション状態の初期化 ---
    def initialize_session_state():
        if "query_handler" not in st.session_state:
            llm, embeddings = get_llm_and_embeddings_cached()
            st.session_state.query_handler = QueryHandler(
                graph=get_neo4j_graph(), llm=llm, embeddings=embeddings
            )
        if "messages" not in st.session_state:
            st.session_state.messages = []

    # --- データ投入ロジック ---
    def handle_ingestion():
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

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                split_documents = text_splitter.split_documents(documents)
                st.write(f"✓ {len(documents)}個のドキュメントを{len(split_documents)}個のチャンクに分割しました。")

                llm, embeddings = get_llm_and_embeddings_cached()
                ingestor = DataIngestor(
                    graph=get_neo4j_graph(), llm=llm, embeddings=embeddings
                )

                status.update(label=f"ステップ 1/4: {len(split_documents)}個のチャンクをグラフ構造に変換しています...")
                
                graph_documents = []
                progress_bar = st.progress(0, text="チャンクの処理状況")
                
                for i, doc in enumerate(split_documents):
                    graph_doc = ingestor.process_single_document(doc)
                    graph_documents.append(graph_doc)
                    progress = (i + 1) / len(split_documents)
                    progress_bar.progress(progress, text=f"チャンク {i + 1}/{len(split_documents)} を処理中")

                st.write(f"✓ {len(split_documents)}個のチャンクから{len(graph_documents)}個のグラフドキュメントを生成しました。")

                status.update(label="ステップ 2/4: グラフドキュメントをNeo4jに格納しています...")
                ingestor.store_graph_documents(graph_documents)
                st.write("✓ グラフドキュメントをNeo4jに正常に格納しました。")

                status.update(label="ステップ 3/4: ベクトルインデックスを作成しています... (時間がかかる場合があります)")
                ingestor.create_vector_index()
                st.write("✓ ベクトルインデックスを正常に作成しました。")

                status.update(label="データ投入が完了しました！", state="complete", expanded=False)
                st.sidebar.success(f"Successfully ingested {len(documents)} documents!")

            except Exception as e:
                status.update(label=f"エラーが発生しました: {e}", state="error", expanded=True)
                st.sidebar.error(f"An error occurred during ingestion: {e}")

    # --- チャットロジック ---
    def handle_query(prompt: str):
        st.session_state.messages.append({"role": "user", "content": prompt})

        rag_chain = st.session_state.query_handler.get_full_chain()
        response = rag_chain.invoke({"question": prompt})

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response.get("answer", "申し訳ありません、回答を生成できませんでした。"),
                "graph_data": response.get("graph_data", []),
                "vector_context": response.get("vector_context", []),
            }
        )

    # --- ヘルパー関数 ---
    def format_graph_data(graph_data: list[dict]) -> tuple[list[Node], list[Edge]]:
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

            if source_id not in node_ids:
                nodes.append(Node(id=source_id, label=source_id, size=15))
                node_ids.add(source_id)

            if target_id not in node_ids:
                nodes.append(Node(id=target_id, label=target_id, size=15))
                node_ids.add(target_id)

            try:
                rel_type = relationship.type
            except AttributeError:
                rel_type = str(relationship)

            edges.append(Edge(source=source_id, target=target_id, label=rel_type))

        return nodes, edges


    # --- メインアプリケーション ---
    def main():
        st.title("グラフRAGチャット")

        with st.sidebar:
            st.header("データ投入")
            st.markdown(
                "ローカルの`./data`ディレクトリにあるMarkdownやPDFファイルをNeo4jに投入し、ベクトルインデックスを作成します。"
            )
            if st.button("データを取り込む"):
                handle_ingestion()

        initialize_session_state()

        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
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
                                            height=400 + i,
                                            directed=True,
                                            physics=True,
                                            hierarchical=False,
                                        )
                                        agraph(nodes=nodes, edges=edges, config=config)
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

        if prompt := st.chat_input("質問を入力してください"):
            st.chat_message("user").markdown(prompt)
            with st.spinner("考え中..."):
                handle_query(prompt)
            st.rerun()

    if __name__ == "__main__":
        main()
