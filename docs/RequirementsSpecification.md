承知いたしました。ご提示いただいた設計メモは、Graph RAGシステムを構築する上で非常に的確かつ構造的です。このメモをベースに、Webでの調査結果や具体的なサンプルコードを交えながら、より詳細で実践的な設計ドキュメントとして全体を再構成し、肉付けしました。

---

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
    2.  検索結果に関連する知識グラフのインタラクティブな可視化。
    3.  LLMが生成した回答と対話履歴の表示領域。
*   **非機能要件**: 応答速度や同時接続数に厳しい制約は設けない（プロトタイプフェーズ）。

#### **3. システムアーキテクチャ設計**

本システムは、「データインジェスト（事前処理）」と「検索・応答（リアルタイム処理）」の2つの主要なフェーズで構成されます。

**3.1. 技術スタック**

*   **グラフデータベース & ベクトルストア**: Neo4j (Community or AuraDB)
*   **LLMアプリケーションフレームワーク**: LangChain
*   **大規模言語モデル (LLM)**: OpenAI GPT-4, GPT-3.5-Turbo など
*   **埋め込みモデル**: OpenAI text-embedding-ada-02, intfloat/multilingual-e5-large など
*   **フロントエンド**: Streamlit
*   **グラフ可視化**: Pyvis, streamlit-agraph

**3.2. 処理フロー概要**

1.  **データインジェスト (バッチ処理)**
    *   **① データ収集・前処理**: PDFやMarkdown文書を読み込み、意味のある単位（チャンク）に分割します。
    *   **② ナレッジグラフ構築**: 各チャンクからLLMを用いてエンティティと関係性を抽出し、Neo4jに知識グラフとして格納します。この際、元のチャンクも`Document`ノードとしてグラフに統合します。
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
    3.  `LLMGraphTransformer`を使い、各チャンクからノードと関係性を抽出します。高性能なLLM（例: GPT-4）ほど抽出の質が向上します。[blog.langchain.com]
    4.  `graph.add_graph_documents`でNeo4jに格納します。`include_source=True`で引用元情報が、`baseEntityLabel=True`で全エンティティへの共通ラベルが付与され、後の処理が容易になります。[blog.langchain.com]

*   **サンプルコード1: グラフ構築**
    ```python
    import os
    from langchain_community.graphs import Neo4jGraph
    from langchain_experimental.graph_transformers import LLMGraphTransformer
    from langchain_openai import ChatOpenAI
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    # --- 1. Neo4jへの接続設定 ---
    os.environ["NEO4J_URI"] = "bolt://localhost:7687"
    os.environ["NEO4J_USERNAME"] = "neo4j"
    os.environ["NEO4J_PASSWORD"] = "your_password"
    os.environ["OPENAI_API_KEY"] = "your_openai_api_key"

    graph = Neo4jGraph()

    # --- 2. ドキュメントの読み込みと分割 ---
    loader = PyPDFLoader("path/to/your/document.pdf")
    documents = loader.load_and_split(
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    )

    # --- 3. グラフ抽出と格納 ---
    # 高性能なモデル（Function Calling対応）が望ましい
    llm = ChatOpenAI(temperature=0, model_name="gpt-4-turbo")
    
    # LLMGraphTransformerの初期化
    llm_transformer = LLMGraphTransformer(llm=llm)

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
    1.  `Neo4jVector.from_existing_graph`メソッドを利用します。このメソッドは非常に強力で、以下の処理を自動で行います。[blog.langchain.com]
        *   指定したノード（ここでは`:Document`）に対するベクトルインデックスを作成。
        *   指定したプロパティ（`text`）の内容を埋め込みモデルでベクトル化。
        *   キーワード検索用の全文検索インデックスも同時に作成。

*   **サンプルコード2: インデックス構築**
    ```python
    from langchain_community.vectorstores import Neo4jVector
    from langchain_openai import OpenAIEmbeddings

    # --- ベクトルストアとインデックスの作成 ---
    # 埋め込みモデルの選択
    embeddings = OpenAIEmbeddings()

    # 既存のグラフからベクトルストアを初期化（インデックスも自動作成）
    vector_store = Neo4jVector.from_existing_graph(
        embedding=embeddings,
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
    1.  ユーザーの質問から、`LLM`の構造化出力機能などを用いて重要なエンティティを抽出します。
    2.  **ベクトル検索**: 質問全体をベクトル化し、`vector_store.similarity_search`で関連性の高い`Document`チャンクを取得します。
    3.  **グラフ検索**: 抽出したエンティティを起点に、Cypherクエリで関連情報を探索します。例えば、そのエンティティに言及している他の文書や、直接関連する他のエンティティを探します。[blog.langchain.com]
    4.  両方の検索結果を統合し、構造化されたコンテキストとしてプロンプトにまとめ、最終的な回答を生成させます。

*   **サンプルコード3: 検索と応答生成チェーン**
    ```python
    from langchain.chains import GraphCypherQAChain
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser

    # (上記で初期化した graph, llm, vector_store を利用)

    # --- 1. グラフ検索用のチェーン ---
    cypher_chain = GraphCypherQAChain.from_llm(
        graph=graph,
        llm=llm,
        verbose=True # 実行されるCypherクエリを確認できる
    )

    # --- 2. ベクトル検索用のリトリーバー ---
    retriever = vector_store.as_retriever()

    # --- 3. プロンプトテンプレート ---
    prompt_template = PromptTemplate.from_template(
        """
あなたは社内文書や学術論文に詳しいアシスタントです。
以下のコンテキスト情報を利用して、質問に答えてください。

**ベクトル検索の結果 (文書の断片):**
{vector_context}

**グラフ検索の結果 (関連エンティティや関係):**
{graph_context}

**質問:**
{question}

**回答:**
"""
    )
    
    # --- 4. 全体を統合したチェーン ---
    def retrieve_context(question: str):
        vector_context = retriever.invoke(question)
        graph_context = cypher_chain.invoke({"query": question})
        return {
            "vector_context": vector_context,
            "graph_context": graph_context.get("result", ""),
            "question": question
        }

    rag_chain = (
        RunnablePassthrough.assign(context=lambda x: retrieve_context(x["question"]))
        | prompt_template
        | llm
        | StrOutputParser()
    )

    # --- 5. チェーンの実行 ---
    question = "LangChainのLLMGraphTransformerについて教えてください。"
    answer = rag_chain.invoke({"question": question})
    print(answer)
    ```

**4.4. フェーズ4: UI (Streamlit) の実装**

ユーザーとの対話を実現するフロントエンドを構築します。

*   **手順**:
    1.  `st.chat_input`と`st.chat_message`を使い、基本的なチャットUIを実装します。[medium.com]
    2.  セッション状態（`st.session_state`）を利用して、会話履歴を保持します。
    3.  バックエンドのRAGチェーンを呼び出し、回答を表示します。
    4.  グラフ可視化には`streamlit-agraph`などのライブラリを使い、検索結果に関連するグラフの一部をインタラクティブに表示します。

*   **サンプルコード4: Streamlit UIの骨格**
    ```python
    import streamlit as st
    from streamlit_agraph import agraph, Node, Edge, Config

    # (上記の rag_chain と graph をバックエンドとして利用)

    st.title("グラフRAGシステム - 文書検索チャット")

    # セッション状態でチャット履歴を管理
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 履歴の表示
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ユーザーからの入力
    if prompt := st.chat_input("質問を入力してください"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("考え中..."):
                # バックエンドのRAGチェーンを呼び出し
                response = rag_chain.invoke({"question": prompt})
                st.markdown(response)

                # 関連グラフの可視化 (簡易版)
                # 実際には、responseからエンティティを抽出し、その近傍を可視化する
                try:
                    cypher_result = graph.query(
                        "MATCH (n)-[r]-(m) WHERE n.id CONTAINS $query RETURN n,r,m LIMIT 10",
                        {"query": prompt.split()[0]} # クエリの最初の単語で検索
                    )
                    
                    nodes = []
                    edges = []
                    node_ids = set()
                    
                    for record in cypher_result:
                        if record["n"]["id"] not in node_ids:
                            nodes.append(Node(id=record["n"]["id"], label=record["n"]["id"], size=15))
                            node_ids.add(record["n"]["id"])
                        if record["m"]["id"] not in node_ids:
                            nodes.append(Node(id=record["m"]["id"], label=record["m"]["id"], size=15))
                            node_ids.add(record["m"]["id"])
                        
                        edges.append(Edge(source=record["n"]["id"], target=record["m"]["id"], label=record["r"].type))

                    if nodes:
                        config = Config(width=750, height=300, directed=True, physics=True, hierarchical=False)
                        agraph(nodes=nodes, edges=edges, config=config)
                except Exception as e:
                    st.error(f"グラフの可視化中にエラーが発生しました: {e}")

        st.session_state.messages.append({"role": "assistant", "content": response})

    ```

#### **5. まとめ**

本設計書で定義したシステムは、Neo4jとLangChainの強力な機能を組み合わせることで、従来の検索システムでは困難だった文脈的な情報探索を実現します。文書から半自動で構築された知識グラフは、データの関係性を可視化・分析するための強力な基盤となり、ベクトル検索とのハイブリッドアプローチにより、検索精度と回答品質の向上が期待できます。
