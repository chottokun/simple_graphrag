🚀 柔軟型ステップアップ運用案
① フリーモード（完全自由抽出）

LLMGraphTransformer を制約なしで利用

想定外のノードやリレーションも含めて抽出させる

Neo4j に投入して、実際にどんな知識グラフになるか観察

👉 この段階で「どんなノードが多いか」「使える関係は何か」を分析します。

② 分析・パターン抽出

抽出されたノードとリレーションを集計し、出現頻度をランキング

MATCH (n) RETURN labels(n), count(*) ORDER BY count(*) DESC
MATCH ()-[r]->() RETURN type(r), count(*) ORDER BY count(*) DESC


実際の利用ケースに照らして「これは有効」「これはノイズ」と仕分け

この結果を YAML/JSON に保存しておく

③ 制約の導入（緩やか）

よく出てくるノードやリレーションを allowed_nodes, allowed_relationships に反映

まだ完全に固定せず、あまりに外れ値が出るのを防ぐ程度の制約にとどめる

④ 制約の強化（本格運用）

確定した業務ドメインのスキーマを設定ファイルとして固定化

プログラムで LLMGraphTransformer に読み込ませる

必要に応じて「禁止リレーション」なども導入

了解です 🙌
では、「Neo4j に投入したグラフからスキーマ候補を自動抽出 → YAML に出力」するシンプルな **スキーマ抽出ツール** の例をご紹介します。

---

## 🛠 Python サンプルコード

```python
from neo4j import GraphDatabase
import yaml

# ===== Neo4j 接続設定 =====
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "password"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

def get_node_labels(tx):
    query = """
    MATCH (n)
    RETURN labels(n) as labels, count(*) as freq
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
        "nodes": [n["labels"][0] for n in nodes],  # 複数ラベルは先頭のみ利用
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
```

---

## 📄 出力される YAML 例

```yaml
nodes:
  - Person
  - Organization
  - Location
  - Event
  - Other
relationships:
  - WORKS_AT
  - LOCATED_IN
  - ATTENDED
  - RELATED_TO
```

---

## ✅ 運用イメージ

1. **自由抽出モード**で生成したグラフを Neo4j に入れる
2. 上記ツールを実行して、出現したノードとリレーションの一覧を `schema.yaml` に保存
3. 研究者/運用者が YAML をレビューして「残す/削除/リネーム」する
4. 更新した YAML を次の `LLMGraphTransformer` の `allowed_nodes` / `allowed_relationships` に渡す

---

💡 これを CI/CD パイプラインに組み込めば、「新しいデータを ingest → スキーマ候補抽出 → 人手レビュー → 本番反映」が流れるようにできます。

---

はい！🙌
では「**Neo4j → スキーマ抽出 → YAML保存 → LLMGraphTransformerに自動反映**」の一連フローをサンプルとして紹介します。

---

# 📦 サンプル構成イメージ

```
project/
 ├─ schema_extractor.py   # Neo4j からスキーマ候補抽出 & YAML 出力
 ├─ transformer_config.py # YAML を読み込んで LLMGraphTransformer に反映
 ├─ schema.yaml           # 抽出されたスキーマ（人手レビュー可）
 └─ main.py               # 全体を流すエントリポイント
```

---

# 1️⃣ `schema_extractor.py`

Neo4j からスキーマ候補を抽出 → `schema.yaml` に保存

```python
from neo4j import GraphDatabase
import yaml

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "password"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

def get_node_labels(tx):
    query = """
    MATCH (n)
    RETURN labels(n) as labels, count(*) as freq
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
        "nodes": [n["labels"][0] for n in nodes],
        "relationships": [r["rel"] for r in rels]
    }
    return schema

def save_schema_to_yaml(schema, path="schema.yaml"):
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(schema, f, allow_unicode=True, sort_keys=False)

if __name__ == "__main__":
    schema = extract_schema()
    print("抽出したスキーマ候補:", schema)
    save_schema_to_yaml(schema)
    print("schema.yaml に保存しました")
```

---

# 2️⃣ `transformer_config.py`

YAML を読み込み、`LLMGraphTransformer` に反映する

```python
import yaml
from graphrag.llm_transformer import LLMGraphTransformer  # simple_graphrag 内の Transformer

def load_schema(path="schema.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def create_transformer_from_schema(schema):
    transformer = LLMGraphTransformer(
        allowed_nodes=schema.get("nodes", []),
        allowed_relationships=schema.get("relationships", [])
    )
    return transformer

if __name__ == "__main__":
    schema = load_schema()
    print("読み込んだスキーマ:", schema)

    transformer = create_transformer_from_schema(schema)
    print("LLMGraphTransformer を構築しました:", transformer)
```

---

# 3️⃣ `main.py`

全体のフローを自動化

```python
import schema_extractor
from transformer_config import load_schema, create_transformer_from_schema

if __name__ == "__main__":
    # スキーマ抽出 → YAML 保存
    schema = schema_extractor.extract_schema()
    schema_extractor.save_schema_to_yaml(schema, "schema.yaml")

    # YAML をロードして Transformer 構築
    schema = load_schema("schema.yaml")
    transformer = create_transformer_from_schema(schema)

    # transformer を使って文書からグラフ抽出
    text = "Elon Musk is the CEO of Tesla, which is headquartered in Austin."
    graph = transformer.transform(text)

    print("抽出されたグラフ:")
    print(graph)
```

---

# 4️⃣ 運用フロー

1. **自由抽出モード**で最初にグラフを作成 → Neo4j に投入
2. `schema_extractor.py` で **出現したノード/関係の候補を YAML に保存**
3. 研究者/運用者が `schema.yaml` を編集して「残す/削除/リネーム」
4. `main.py` で **制約付き Transformer** を自動構築して次回の抽出に利用

---

✅ これにより、

* 最初は「自由抽出」で発見的に
* 次第に「制約付き抽出」で精度と一貫性を高める

という「柔軟型（free → restrict）」の運用が可能になります。

---


