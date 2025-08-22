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

```python
from neo4j import GraphDatabase
import yaml
import os
from src.config import get_neo4j_credentials, load_app_config

# アプリケーション設定のロード (環境変数からNeo4j認証情報を取得するため)
load_app_config()

# Neo4j 接続設定
neo4j_creds = get_neo4j_credentials()
NEO4J_URI = neo4j_creds["uri"]
NEO4J_USER = neo4j_creds["username"]
NEO4J_PASS = neo4j_creds["password"]

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

def get_node_labels(tx):
    query = """
    MATCH (n)
    UNWIND labels(n) as label
    RETURN label, count(*) as freq
    ORDER BY freq DESC
    """
    return [dict(record) for record in tx.run(query)]

def get_relationship_types(tx):
    query = """
    MATCH ()-[r]->()
    RETURN type(r) as rel, count(*) as freq
    ORDER BY freq DESC
    """
    return [dict(record) for record in tx.run(query)]

def extract_schema():
    with driver.session() as session:
        nodes = session.execute_read(get_node_labels)
        rels = session.execute_read(get_relationship_types)

    schema = {
        "nodes": [n["label"] for n in nodes],
        "relationships": [r["rel"] for r in rels]
    }
    return schema

def save_schema_to_yaml(schema, path="schema.yaml"):
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(schema, f, allow_unicode=True, sort_keys=False)

if __name__ == "__main__":
    schema = extract_schema()
    print("抽出したスキーマ候補:")
    print(schema)

    save_schema_to_yaml(schema, "schema.yaml")
    print("schema.yaml に保存しました。")

    # 補足: Neo4jドライバーのクリーンアップ
    # スクリプトの終了時にドライバーを閉じることで、リソースリークを防ぎます。
    driver.close()
```

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