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