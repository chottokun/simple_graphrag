# 高度なグラフRAGシステム (Advanced GraphRAG System)

これは、`Neo4j`、`LangChain`、`Ollama` を使用して、高度な検索拡張生成（RAG）チャットシステムを実装するプロジェクトです。ベクトル検索と知識グラフ検索を組み合わせたハイブリッドアプローチを用いて、**アップロードされたドキュメント**から文脈的に豊かな回答を生成します。

フロントエンドは `Streamlit` で構築されており、UIから直接ドキュメントの取り込みとチャットが可能です。

## 主な機能

- **ドキュメント取り込み:**
  - Streamlit UIのサイドバーから、Markdown (`.md`) や PDF (`.pdf`) ファイルをアップロードできます。
  - アップロードされたドキュメントはチャンクに分割され、ベクトル埋め込みと共にNeo4jデータベースに取り込まれます。
- **ハイブリッド検索によるQ&A:**
  - ユーザーの質問に対して、まずNeo4jのベクトルインデックスを利用して関連性の高いドキュメントチャンクを検索します。
  - 検索結果をコンテキストとしてLLMに提供し、ドキュメントに基づいた正確な回答を生成します。

## プロジェクト構造

- `app.py`: Streamlitアプリケーションのメインファイル。UIとバックエンドハンドラを含みます。
- `src/`: コアとなるビジネスロジック。
  - `config.py`: 環境変数の読み込みと設定管理。
  - `data_ingestion.py`: ドキュメントの処理、知識グラフの構築、ベクトルインデックスの作成を行う `DataIngestor` クラス。
  - `query_handler.py`: ユーザーのクエリを処理し、ハイブリッド検索を実行して回答を生成する `QueryHandler` クラス。
- `tests/`: `pytest` を用いたユニットテスト。
- `data/`: 取り込み対象のドキュメントを格納するディレクトリ（Git管理外）。
- `neo4j-docker-compose.yaml`: Neo4jを簡単に起動するためのDocker Composeファイル。
- `docs/`: 設計書や要件定義書。

## セットアップ方法

1.  **リポジトリのクローン:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **環境変数の設定:**
    `.env.example` ファイルをコピーして `.env` という名前のファイルを作成します。
    ```bash
    cp .env.example .env
    ```
    その後、`.env` ファイルを編集し、お使いの環境に合わせて接続情報を設定してください。Docker Composeを使用する場合、Neo4jのデフォルト値は既に設定されています。
    - `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`
    - `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_EMBEDDING_MODEL`

3.  **依存関係のインストール:**
    プロジェクトのルートにある `requirements.txt` を使用して、必要なPythonライブラリをインストールします。
    ```bash
    pip install -r requirements.txt
    ```

## 実行方法

### 1. 外部サービスの起動

このアプリケーションを実行する前に、以下の外部サービスが起動していることを確認してください。

- **Neo4j データベース:**
  プロジェクトルートにある `neo4j-docker-compose.yaml` を使って、Dockerコンテナとして簡単に起動できます。
  ```bash
  docker-compose -f neo4j-docker-compose.yaml up -d
  ```

- **Ollama サーバー:**
  Ollamaサーバーを別途起動し、必要なモデルをプルしておきます。
  ```bash
  # モデルのプル
  ollama pull gemma:2b-instruct-q4_0
  ollama pull mxbai-embed-large
  ```

### 2. Streamlitアプリケーションの実行

プロジェクトのルートディレクトリで以下のコマンドを実行します。
```bash
streamlit run app.py
```
ブラウザでStreamlitのUIが開きます。

### 3. データインジェストの実行

1.  `data/` ディレクトリを作成し、その中に取り込みたいMarkdownまたはPDFファイルを置きます。
2.  Streamlitアプリケーションのサイドバーにある「**Ingest Data**」ボタンをクリックします。
3.  処理が完了すると、データベースにデータが取り込まれ、チャットでその内容について質問できるようになります。

## テスト

テストは外部サービスへの実際の接続を必要としない、モックベースのテストです。

テストスイート全体を実行するには、以下のコマンドを実行します。
```bash
pytest
```