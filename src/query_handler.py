from langchain_community.vectorstores import Neo4jVector
from langchain_neo4j import Neo4jGraph
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings
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
            self.schema = self.graph.get_schema()
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

    def get_entity_extraction_chain(self) -> Runnable:
        """
        Creates a chain to extract key entities from a question.
        """
        entity_extraction_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert at extracting key entities, keywords, and proper nouns from a question. "
                    "Extract a list of entities that are relevant for a graph database query. "
                    "Return the entities as a comma-separated list. Do not add any explanation or apologies.",
                ),
                ("human", "Question: {question}"),
            ]
        )

        entity_extraction_chain = (
            entity_extraction_prompt | self.llm | CommaSeparatedListOutputParser()
        )
        return entity_extraction_chain

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
        entity_chain = self.get_entity_extraction_chain()
        graph_query_template = """
            MATCH (n)-[r]-(m)
            WHERE n.id IN $entities OR m.id IN $entities
            RETURN n, r, m
            LIMIT 20
        """

        def retrieve_all_data(inputs: dict) -> dict:
            """
            Extracts entities from the question and uses them to query the graph
            with a safe, templated query.
            """
            question = inputs["question"]
            try:
                entities = entity_chain.invoke({"question": question})
            except Exception as e:
                print(f"Entity extraction failed: {e}")
                entities = []

            vector_context = vector_retriever.invoke(question)
            try:
                graph_data = self.graph.query(
                    graph_query_template, params={"entities": entities}
                )
            except Exception as e:
                print(f"Templated graph query failed: {e}")
                graph_data = []

            return {
                "question": question,
                "vector_context": vector_context,
                "graph_context": graph_data,
                "graph_data_for_viz": graph_data,
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
