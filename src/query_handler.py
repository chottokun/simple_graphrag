from langchain_community.vectorstores import Neo4jVector
from langchain_neo4j import Neo4jGraph
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
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

    def get_cypher_generation_chain(self) -> Runnable:
        """
        Creates a chain to generate a Cypher query from a question.
        """
        cypher_generation_system_template = """
        You are an expert Neo4j developer who writes Cypher based on a user's request.
        Given a question, you need to write a Cypher query that can retrieve relevant information from the database.
        The query should be as simple as possible and should not contain any explanations or apologies.

        Schema:
        {schema}

        Question: {question}
        Cypher Query:
        """
        cypher_generation_prompt = ChatPromptTemplate.from_messages([
            ("system", cypher_generation_system_template),
            ("human", "{question}")
        ])

        cypher_generation_chain = (
            RunnablePassthrough.assign(
                schema=lambda _: self.schema
            )
            | cypher_generation_prompt
            | self.llm.bind(stop=["\nCypher:"])
            | StrOutputParser()
        )
        return cypher_generation_chain

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
        cypher_chain = self.get_cypher_generation_chain()

        def retrieve_all_data(inputs: dict) -> dict:
            """
            Fetches all necessary data: vector context and graph context.
            It preserves the raw graph data for later visualization.
            """
            question = inputs["question"]
            generated_cypher = cypher_chain.invoke({"question": question})
            vector_context = vector_retriever.invoke(question)
            try:
                graph_data = self.graph.query(generated_cypher)
            except Exception as e:
                print(f"Graph query failed: {e}")
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
                }
            )
        )
        return chain
