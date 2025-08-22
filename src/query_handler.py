from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
import re
import re
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser
from operator import itemgetter


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
        print("QueryHandler __init__ called.")
        try:
            self.schema = self.graph.schema
            print(f"Schema set: {self.schema[:50]}...")
        except Exception as e:
            print(f"Error setting schema: {e}")
            self.schema = "" # Set to empty string to avoid AttributeError

    def get_vector_retriever(self) -> Runnable:
        """
        Creates and returns a vector retriever from the Neo4j index.
        """
        vector_store = Neo4jVector(
            embedding=self.embeddings,
            graph=self.graph
        )
        return vector_store.as_retriever()

    def get_full_chain(self) -> Runnable:
        """
        Assembles and returns the full RAG chain, which now returns both
        the text answer and the graph data for visualization.
        """
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

        vector_retriever = self.get_vector_retriever()

        # Initialize GraphCypherQAChain
        cypher_qa_chain = GraphCypherQAChain.from_llm(
            graph=self.graph,
            llm=self.llm,
            verbose=True  # Set to True for debugging to see generated Cypher queries
        )

        def retrieve_all_data(inputs: dict) -> dict:
            """
            Extracts entities from the question and uses them to query the graph
            with a safe, templated query.
            """
            question = inputs["question"]

            vector_context = vector_retriever.invoke(question)
            
            # Use GraphCypherQAChain to get graph context and data
            try:
                cypher_qa_result = cypher_qa_chain.invoke({"query": question})
                graph_context = cypher_qa_result.get("result", "")
                # GraphCypherQAChain returns the answer, not raw graph data for viz directly.
                # For visualization, we might need to parse the generated Cypher query or
                # execute a separate query to get the graph data based on the entities identified by the chain.
                # For now, we'll just pass the result as graph_context.
                # A more advanced implementation would involve extracting the Cypher query from the chain's output
                # and executing it to get the graph data for visualization.
                graph_data_for_viz = graph_context # Placeholder, needs refinement for actual graph data
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

        answer_chain = (
            itemgetter("context")
            | prompt_template
            | self.llm
            | StrOutputParser()
        )

        chain = (
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
        return chain
