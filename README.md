# 高度なグラフRAGシステム (Advanced GraphRAG System)

これは、`Neo4j`、`LangChain`、`Ollama` を使用して、高度な検索拡張生成（RAG）チャットシステムを実装するプロジェクトです。ベクトル検索と知識グラフ検索を組み合わせたハイブリッドアプローチを用いて、ドキュメントから文脈的に豊かな回答を生成します。

フロントエンドは `Streamlit` で構築されています。

## プロジェクト構造

- `app.py`: Streamlitアプリケーションのメインファイル。UIとバックエンドハンドラを含みます。
- `src/`: コアとなるビジネスロジック。
  - `config.py`: 環境変数の読み込みと設定管理。
  - `data_ingestion.py`: ドキュメントの処理、知識グラフの構築、ベクトルインデックスの作成を行う `DataIngestor` クラス。
  - `query_handler.py`: ユーザーのクエリを処理し、ハイブリッド検索を実行して回答を生成する `QueryHandler` クラス。
- `tests/`: `pytest` を用いたユニットテスト。
  - `conftest.py`: グローバルなテスト設定（`.env`の読み込みなど）。
  - 各モジュールに対応するテストファイル。
- `docs/`: 設計書や要件定義書。

## セットアップ方法

1.  **リポジトリのクローン:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **依存関係のインストール:**
    プロジェクトのルートにある `requirements.txt` を使用して、必要なPythonライブラリをインストールします。
    ```bash
    pip install -r requirements.txt
    ```

3.  **環境変数の設定:**
    `.env.example` ファイルをコピーして `.env` という名前のファイルを作成します。
    ```bash
    cp .env.example .env
    ```
    その後、`.env` ファイルを編集し、お使いの環境に合わせて `Neo4j` と `Ollama` の接続情報を設定してください。
    - `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`
    - `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_EMBEDDING_MODEL`

## 実行方法

### 1. 外部サービスの起動

このアプリケーションを実行する前に、以下の外部サービスが起動していることを確認してください。
- **Neo4j データベース**
- **Ollama サーバー** (`gemma3:4b-it-qat` と `mxbai-embed-large` のモデルが利用可能であること)

### 2. Streamlitアプリケーションの実行

プロジェクトのルートディレクトリで以下のコマンドを実行します。
```bash
streamlit run app.py
```
ブラウザでStreamlitのUIが開きます。

### 3. (オプション) データインジェストの実行
現状、データインジェストを実行するための独立したスクリプトはありません。`DataIngestor`クラスのメソッドを呼び出すPythonスクリプトを別途作成する必要があります。

## テスト

テストは外部サービスへの実際の接続を必要としない、モックベースのテストです。

テストスイート全体を実行するには、以下のコマンドを実行します。
```bash
pytest
```
