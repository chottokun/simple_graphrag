# スキーマ管理 (Schema Management)

本ドキュメントでは、`simple_graphrag`における知識グラフのスキーマを柔軟に管理し、`LLMGraphTransformer`のエンティティ抽出精度を向上させるための運用案とツールについて説明します。

## 🚀 柔軟型ステップアップ運用案

### ① フリーモード（完全自由抽出）

`LLMGraphTransformer` を制約なしで利用し、想定外のノードやリレーションも含めて抽出させます。Neo4j に投入して、実際にどんな知識グラフになるか観察します。

👉 この段階で「どんなノードが多いか」「使える関係は何か」を分析します。

### ② 分析・パターン抽出

抽出されたノードとリレーションを集計し、出現頻度をランキングします。

```cypher
MATCH (n) RETURN labels(n) as labels, count(*) as freq ORDER BY freq DESC
MATCH ()-[r]->() RETURN type(r) as rel, count(*) as freq ORDER BY freq DESC
```

実際の利用ケースに照らして「これは有効」「これはノイズ」と仕分けし、この結果を YAML/JSON に保存します。

### ③ 制約の導入（緩やか）

よく出てくるノードやリレーションを `GRAPH_ALLOWED_NODES`, `GRAPH_ALLOWED_RELATIONSHIPS` 環境変数に反映します。まだ完全に固定せず、あまりに外れ値が出るのを防ぐ程度の制約にとどめます。

### ④ 制約の強化（本格運用）

確定した業務ドメインのスキーマを環境変数として固定化し、プログラムで `LLMGraphTransformer` に読み込ませます。必要に応じて「禁止リレーション」なども検討します。

---

## 🛠 スキーマ抽出ツール

Neo4j に投入したグラフからスキーマ候補を自動抽出し、YAML に出力するシンプルなツールです。

**補足**: 本サンプルコードは簡略化されており、実運用ではNeo4j接続エラーやクエリ実行エラーに対する堅牢なエラーハンドリングを追加することを推奨します。

# スキーマ管理 (Schema Management)

本ドキュメントでは、`simple_graphrag`における知識グラフのスキーマを柔軟に管理し、`LLMGraphTransformer`のエンティティ抽出精度を向上させるための運用案とツールについて説明します。

## 🚀 柔軟型ステップアップ運用案

### ① フリーモード（完全自由抽出）

`LLMGraphTransformer` を制約なしで利用し、想定外のノードやリレーションも含めて抽出させます。Neo4j に投入して、実際にどんな知識グラフになるか観察します。

👉 この段階で「どんなノードが多いか」「使える関係は何か」を分析します。

### ② 分析・パターン抽出

抽出されたノードとリレーションを集計し、出現頻度をランキングします。

```cypher
MATCH (n) RETURN labels(n) as labels, count(*) as freq ORDER BY freq DESC
MATCH ()-[r]->() RETURN type(r) as rel, count(*) as freq ORDER BY freq DESC
```

実際の利用ケースに照らして「これは有効」「これはノイズ」と仕分けし、この結果を YAML/JSON に保存します。

### ③ 制約の導入（緩やか）

よく出てくるノードやリレーションを `GRAPH_ALLOWED_NODES`, `GRAPH_ALLOWED_RELATIONSHIPS` 環境変数に反映します。まだ完全に固定せず、あまりに外れ値が出るのを防ぐ程度の制約にとどめます。

### ④ 制約の強化（本格運用）

確定した業務ドメインのスキーマを環境変数として固定化し、プログラムで `LLMGraphTransformer` に読み込ませます。必要に応じて「禁止リレーション」なども検討します。

---

## 🛠 スキーマ抽出ツール (`schema_extractor.py`)

Neo4j に投入したグラフからスキーマ候補を自動抽出し、YAML に出力するシンプルなツールです。

### スクリプトの解説

```python
from neo4j import GraphDatabase, exceptions
import yaml
import os
from src.config import get_neo4j_credentials, load_app_config

# アプリケーション設定のロード (環境変数からNeo4j認証情報を取得するため)
load_app_config()

def extract_schema(driver):
    """Neo4jからノードラベルと関係タイプを抽出する"""
    try:
        with driver.session() as session:
            # ノードラベルの取得
            nodes_query = "MATCH (n) UNWIND labels(n) as label RETURN label, count(*) as freq ORDER BY freq DESC"
            nodes_result = session.run(nodes_query)
            nodes = [record["label"] for record in nodes_result]

            # 関係タイプの取得
            rels_query = "MATCH ()-[r]->() RETURN type(r) as rel, count(*) as freq ORDER BY freq DESC"
            rels_result = session.run(rels_query)
            rels = [record["rel"] for record in rels_result]

        return {"nodes": nodes, "relationships": rels}
    except exceptions.ServiceUnavailable as e:
        print(f"[エラー] Neo4jデータベースに接続できません。URI、認証情報を確認してください。詳細: {e}")
        return None
    except exceptions.AuthError as e:
        print(f"[エラー] Neo4jの認証に失敗しました。ユーザー名とパスワードを確認してください。詳細: {e}")
        return None
    except Exception as e:
        print(f"[エラー] スキーマ抽出中に予期せぬエラーが発生しました: {e}")
        return None

def save_schema_to_yaml(schema, path="schema.yaml"):
    """スキーマをYAMLファイルに保存する"""
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(schema, f, allow_unicode=True, sort_keys=False)

if __name__ == "__main__":
    driver = None
    try:
        # Neo4j 接続設定
        neo4j_creds = get_neo4j_credentials()
        uri = neo4j_creds["uri"]
        user = neo4j_creds["username"]
        password = neo4j_creds["password"]

        # ドライバーの初期化
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        print("Neo4jに正常に接続しました。")

        schema = extract_schema(driver)

        if schema:
            print("\n抽出したスキーマ候補:")
            print(schema)
            save_schema_to_yaml(schema, "schema.yaml")
            print("\nschema.yaml に保存しました。")

    finally:
        if driver:
            driver.close()
            print("\nNeo4jドライバーをクローズしました。")
```

### トラブルシューティング

スクリプト実行時に発生する可能性のある一般的なエラーと対処法です。

*   **`[エラー] Neo4jデータベースに接続できません。`**
    *   **原因**: Neo4jサーバーが起動していない、または `.env` ファイルの `NEO4J_URI` が正しくありません。
    *   **対処法**: Dockerコンテナ（`docker-compose up`）やNeo4j Desktopが起動していることを確認してください。また、URI（例: `bolt://localhost:7687`）が正しいか確認してください。

*   **`[エラー] Neo4jの認証に失敗しました。`**
    *   **原因**: `.env` ファイルの `NEO4J_USERNAME` または `NEO4J_PASSWORD` が間違っています。
    *   **対処法**: Neo4j Browserにログインできる認証情報と一致しているか確認してください。

*   **`[エラー] スキーマ抽出中に予期せぬエラーが発生しました`**
    *   **原因**: Cypherクエリの構文エラーや、データベースの一時的な問題などが考えられます。
    *   **対処法**: 表示されたエラーメッセージを元に、クエリやデータベースの状態を確認してください。

---

## 📄 出力される YAML 例

```yaml
nodes:
  - Person
  - Organization
  - Location
  - Event
  - Document
  - Other
relationships:
  - WORKS_AT
  - LOCATED_IN
  - ATTENDED
  - RELATED_TO
  - MENTIONS
```

---

## ✅ 運用イメージ

1.  **自由抽出モード**で生成したグラフを Neo4j に投入します。
2.  `python schema_extractor.py` を実行して、出現したノードとリレーションの一覧を `schema.yaml` に保存します。
3.  研究者/運用者が `schema.yaml` をレビューし、「残す/削除/リネーム」などの編集を行います。
4.  更新した `schema.yaml` の内容を、`.env` ファイルの `GRAPH_ALLOWED_NODES` および `GRAPH_ALLOWED_RELATIONSHIPS` 環境変数に設定します。

    例えば、`schema.yaml` の内容が以下のようだった場合：
    ```yaml
nodes:
  - Person
  - Organization
  - Location
relationships:
  - WORKS_AT
  - LOCATED_IN
```
    `.env` ファイルには以下のように設定します：
    ```
    GRAPH_ALLOWED_NODES="Person,Organization,Location"
    GRAPH_ALLOWED_RELATIONSHIPS="WORKS_AT,LOCATED_IN"
    ```
    これにより、`LLMGraphTransformer` はこれらの制約に従ってエンティティを抽出します。

---

💡 これを CI/CD パイプラインに組み込めば、「新しいデータを ingest → スキーマ候補抽出 → 人手レビュー → 環境変数更新 → 本番反映」が流れるようにできます。

---

✅ これにより、

*   最初は「自由抽出」で発見的に
*   次第に「制約付き抽出」で精度と一貫性を高める

という「柔軟型（free → restrict）」の運用が可能になります。

---

## 補足: `schema_extractor.py` について

リポジトリのルートに配置されている `schema_extractor.py` は、上記で説明したスキーマ抽出ツールの実装です。このスクリプトを実行することで、現在のNeo4jデータベースからスキーマ情報を抽出し、`schema.yaml`として出力できます。

**実行方法:**

```bash
python schema_extractor.py
```

このスクリプトは、`.env`ファイルからNeo4jの接続情報を読み込みますので、実行前に`.env`ファイルが正しく設定されていることを確認してください。


---

## 📄 出力される YAML 例

```yaml
nodes:
  - Person
  - Organization
  - Location
  - Event
  - Document
  - Other
relationships:
  - WORKS_AT
  - LOCATED_IN
  - ATTENDED
  - RELATED_TO
  - MENTIONS
```

---

## ✅ 運用イメージ

1.  **自由抽出モード**で生成したグラフを Neo4j に投入します。
2.  上記の `スキーマ抽出ツール` を実行して、出現したノードとリレーションの一覧を `schema.yaml` に保存します。
3.  研究者/運用者が `schema.yaml` をレビューし、「残す/削除/リネーム」などの編集を行います。
4.  更新した `schema.yaml` の内容を、`GRAPH_ALLOWED_NODES` および `GRAPH_ALLOWED_RELATIONSHIPS` 環境変数に設定します。

    例えば、`schema.yaml` の内容が以下のようだった場合：
    ```yaml
    nodes:
      - Person
      - Organization
      - Location
    relationships:
      - WORKS_AT
      - LOCATED_IN
    ```
    `.env` ファイルには以下のように設定します：
    ```
    GRAPH_ALLOWED_NODES="Person,Organization,Location"
    GRAPH_ALLOWED_RELATIONSHIPS="WORKS_AT,LOCATED_IN"
    ```
    これにより、`LLMGraphTransformer` はこれらの制約に従ってエンティティを抽出します。

---

💡 これを CI/CD パイプラインに組み込めば、「新しいデータを ingest → スキーマ候補抽出 → 人手レビュー → 環境変数更新 → 本番反映」が流れるようにできます。

---

✅ これにより、

*   最初は「自由抽出」で発見的に
*   次第に「制約付き抽出」で精度と一貫性を高める

という「柔軟型（free → restrict）」の運用が可能になります。