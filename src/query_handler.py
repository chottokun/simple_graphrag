from langchain_community.vectorstores import Neo4jVector
from langchain_neo4j import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class QueryHandler:
    """
    Handles user queries by orchestrating a hybrid search
    (vector + graph) and generating a response.
    Designed for dependency injection.
    """

    def __init__(
        self,
        graph: Neo4jGraph,
        llm: BaseLanguageModel,
        embeddings: Embeddings
    ):
        """
        Initializes the QueryHandler with its dependencies.
        """
        self.graph = graph
        self.llm = llm
        self.embeddings = embeddings

    def get_vector_retriever(self) -> Runnable:
        """
        Creates and returns a vector retriever from the Neo4j index.
        """
        vector_store = Neo4jVector(
            embedding=self.embeddings,
            graph=self.graph
        )
        return vector_store.as_retriever()

    def get_graph_cypher_chain(self) -> Runnable:
        """
        Creates and returns a GraphCypherQAChain.
        """
        chain = GraphCypherQAChain.from_llm(
            llm=self.llm,
            graph=self.graph,
            verbose=True
        )
        return chain

    def get_full_chain(self) -> Runnable:
        """
        Assembles and returns the full RAG chain.
        """
        prompt_template = ChatPromptTemplate.from_template(
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

        vector_retriever = self.get_vector_retriever()
        cypher_chain = self.get_graph_cypher_chain()

        def retrieve_context(inputs: dict) -> dict:
            """
            Takes the input dict (with a "question" key) and returns a dict
            with all the necessary context for the prompt.
            """
            question = inputs["question"]

            # Retrieve context from both sources
            vector_context = vector_retriever.invoke(question)
            graph_context = cypher_chain.invoke({"query": question}).get("result", "")

            # Return a dictionary with all keys required by the prompt
            return {
                "vector_context": vector_context,
                "graph_context": graph_context,
                "question": question,
            }

        rag_chain = (
            retrieve_context
            | prompt_template
            | self.llm
            | StrOutputParser()
        )

        return rag_chain
