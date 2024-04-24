from functools import cached_property
import os
from typing import List, Tuple
from dotenv import load_dotenv
import openai
import pandas as pd
from qdrant_client import QdrantClient, grpc, models


from tools.utils import deterministic_uuid

DOCUMENTATION_COLLECTION_NAME = "documentation"
DDL_COLLECTION_NAME = "ddl"
SQL_COLLECTION_NAME = "sql"
SCROLL_SIZE = 1000
EMBEDDING_TYPE="openai"
ID_SUFFIXES = {
    DDL_COLLECTION_NAME: "ddl",
    DOCUMENTATION_COLLECTION_NAME: "doc",
    SQL_COLLECTION_NAME: "sql",
}

VECTOR_SIZE= 1536


class OpenAIEmbeddings:
    def __init__(self, api_key):
        self.api_key = api_key
        # Initialize OpenAI API client
        openai.api_key = api_key

    def generate_embeddings(self, documents):
        # Perform language chain OpenAI embeddings of given documents
        embeddings = []
        for document in documents:
            # Generate embeddings using OpenAI API
            embedding = openai.Embed(document, self.database_url)
            embeddings.append(embedding)
        return embeddings

    def save_embeddings(self, embeddings, output_file):
        # Save the embeddings to an output file
        with open(output_file, 'w') as file:
            for embedding in embeddings:
                file.write(str(embedding) + '\n')


class QdrantEmbeddings(OpenAIEmbeddings):
    def __init__(self, api_key, database_url, vector_database_url):
        super().__init__(api_key, database_url)
        self.vector_database_url = vector_database_url

    def save_to_vector_database(self, embedding, vector_database_url):
        i = 200
        # Save embedding to the vector database using the provided URL
        return False

    def save_embeddings(self, embeddings):
        # Save the embeddings to the vector database
        for embedding in embeddings:
            res = self.save_to_vector_database(embedding, self.vector_database_url)

            # Save embedding to the vector database using the provided URL
            #(embedding, self.vector_database_url)

# Most of the implementation below is a copy or vanna/qdrant/qdrant.py

class qdrant_store:
    def __init__(self, api_key, vector_database_url):
        self.api_key = api_key
        self.vector_database_url = vector_database_url
        '''
            QDRANTClient    Args:
                - client: A `qdrant_client.QdrantClient` instance. Overrides other config options.
                - location: If `":memory:"` - use in-memory Qdrant instance. If `str` - use it as a `url` parameter.
                - url: Either host or str of "Optional[scheme], host, Optional[port], Optional[prefix]". Eg. `"http://localhost:6333"`.
                - prefer_grpc: If `true` - use gPRC interface whenever possible in custom methods.
                - https: If `true` - use HTTPS(SSL) protocol. Default: `None`
                - api_key: API key for authentication in Qdrant Cloud. Default: `None`
                - timeout: Timeout for REST and gRPC API requests. Defaults to 5 seconds for REST and unlimited for gRPC.
                - path: Persistence path for QdrantLocal. Default: `None`.
                - prefix: Prefix to the REST URL paths. Example: `service/v1` will result in `http://localhost:6333/service/v1/{qdrant-endpoint}`.
                - n_results: Number of results to return from similarity search. Defaults to 10.
                - fastembed_model: [Model](https://qdrant.github.io/fastembed/examples/Supported_Models/#supported-text-embedding-models) to use for `fastembed.TextEmbedding`.
                  Defaults to `"BAAI/bge-small-en-v1.5"`.
                - collection_params: Additional parameters to pass to `qdrant_client.QdrantClient#create_collection()` method.
                - distance_metric: Distance metric to use when creating collections. Defaults to `qdrant_client.models.Distance.COSINE`.
        '''        
        self._client = QdrantClient(url=self.vector_database_url)
        #TODO get rid of this hardcoding
        self.embedding_model = "text-embedding-3-small"
        self.distance_metric = models.Distance.COSINE
        self._setup_collections()
        self.n_results = 10


    def add_question_sql(self, question: str, sql: str, **kwargs) -> str:
        #question_answer = format("Question: {0}\n\nSQL: {1}", (question, sql))
        question_answer = "Question:" + question + "\n\nSQL:" + sql 
        id = deterministic_uuid(question_answer)

        self._client.upsert(
            SQL_COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=id,
                    vector=self.generate_embedding(question_answer),
                    payload={
                        "question": question,
                        "sql": sql,
                    },
                )
            ],
        )

        return self._format_point_id(id, SQL_COLLECTION_NAME)

    def add_ddl(self, ddl: str, **kwargs) -> str:
        id = deterministic_uuid(ddl)
        self._client.upsert(
            DDL_COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=id,
                    vector=self.generate_embedding(ddl),
                    payload={
                        "ddl": ddl,
                    },
                )
            ],
        )
        return self._format_point_id(id, DDL_COLLECTION_NAME)

    def add_documentation(self, documentation: str, **kwargs) -> str:
        id = deterministic_uuid(documentation)
        vec = self.generate_embedding(documentation)
        print(type(vec[0]))
        self._client.upsert(
            DOCUMENTATION_COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=id,
                    vector=self.generate_embedding(documentation),
                    payload={
                        "documentation": documentation,
                    },
                )
            ],
        )

        return self._format_point_id(id, DOCUMENTATION_COLLECTION_NAME)
 
    def get_training_data(self, **kwargs) -> pd.DataFrame:
        df = pd.DataFrame()

        if sql_data := self._get_all_points(SQL_COLLECTION_NAME):
            question_list = [data.payload["question"] for data in sql_data]
            sql_list = [data.payload["sql"] for data in sql_data]
            id_list = [
                self._format_point_id(data.id, SQL_COLLECTION_NAME) for data in sql_data
            ]

            df_sql = pd.DataFrame(
                {
                    "id": id_list,
                    "question": question_list,
                    "content": sql_list,
                }
            )

            df_sql["training_data_type"] = "sql"

            df = pd.concat([df, df_sql])

        if ddl_data := self._get_all_points(DDL_COLLECTION_NAME):
            ddl_list = [data.payload["ddl"] for data in ddl_data]
            id_list = [
                self._format_point_id(data.id, DDL_COLLECTION_NAME) for data in sql_data
            ]

            df_ddl = pd.DataFrame(
                {
                    "id": id_list,
                    "question": [None for _ in ddl_list],
                    "content": ddl_list,
                }
            )

            df_ddl["training_data_type"] = "ddl"

            df = pd.concat([df, df_ddl])

        doc_data = self.documentation_collection.get()

        if doc_data := self._get_all_points(DOCUMENTATION_COLLECTION_NAME):
            document_list = [data.payload["documentation"] for data in doc_data]
            id_list = [
                self._format_point_id(data.id, DOCUMENTATION_COLLECTION_NAME)
                for data in doc_data
            ]

            df_doc = pd.DataFrame(
                {
                    "id": id_list,
                    "question": [None for _ in document_list],
                    "content": document_list,
                }
            )

            df_doc["training_data_type"] = "documentation"

            df = pd.concat([df, df_doc])

        return df

    def remove_training_data(self, id: str, **kwargs) -> bool:
        try:
            id, collection_name = self._parse_point_id(id)
            self._client.delete(collection_name, points_selector=[id])
        except ValueError:
            return False

    def remove_collection(self, collection_name: str) -> bool:
        """
        This function can reset the collection to empty state.

        Args:
            collection_name (str): sql or ddl or documentation

        Returns:
            bool: True if collection is deleted, False otherwise
        """
        if collection_name in ID_SUFFIXES.keys():
            self._client.delete_collection(collection_name)
            self._setup_collections()
            return True
        else:
            return False

    @cached_property
    def embeddings_dimension(self):
        return VECTOR_SIZE

    def get_similar_question_sql(self, question: str, **kwargs) -> list:
        results = self._client.search(
            SQL_COLLECTION_NAME,
            query_vector=self.generate_embedding(question),
            limit=self.n_results,
            with_payload=True,
        )

        return [dict(result.payload) for result in results]

    def get_related_ddl(self, question: str, **kwargs) -> list:
        results = self._client.search(
            DDL_COLLECTION_NAME,
            query_vector=self.generate_embedding(question),
            limit=self.n_results,
            with_payload=True,
        )

        return [result.payload["ddl"] for result in results]

    def get_related_documentation(self, question: str, **kwargs) -> list:
        results = self._client.search(
            DOCUMENTATION_COLLECTION_NAME,
            query_vector=self.generate_embedding(question),
            limit=self.n_results,
            with_payload=True,
        )

        return [result.payload["documentation"] for result in results]

    def generate_embedding(self, data: str, **kwargs) -> List[float]:
        embedding = openai.embeddings.create( input = data, model=self.embedding_model)
        return embedding.data[0].embedding

    def _get_all_points(self, collection_name: str):
        results: List[models.Record] = []
        next_offset = None
        stop_scrolling = False
        while not stop_scrolling:
            records, next_offset = self._client.scroll(
                collection_name,
                limit=SCROLL_SIZE,
                offset=next_offset,
                with_payload=True,
                with_vectors=False,
            )
            stop_scrolling = next_offset is None or (
                isinstance(next_offset, grpc.PointId)
                and next_offset.num == 0
                and next_offset.uuid == ""
            )

            results.extend(records)

        return results

    def _setup_collections(self):
        if not self._client.collection_exists(SQL_COLLECTION_NAME):
            self._client.create_collection(
                collection_name=SQL_COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=self.embeddings_dimension,
                    distance=self.distance_metric,
                ),
            )

        if not self._client.collection_exists(DDL_COLLECTION_NAME):
            self._client.create_collection(
                collection_name=DDL_COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=self.embeddings_dimension,
                    distance=self.distance_metric,
                ),
            )
        if not self._client.collection_exists(DOCUMENTATION_COLLECTION_NAME):
            self._client.create_collection(
                collection_name=DOCUMENTATION_COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=self.embeddings_dimension,
                    distance=self.distance_metric,
                ),
            )

    def _format_point_id(self, id: str, collection_name: str) -> str:
        return "{0}-{1}".format(id, ID_SUFFIXES[collection_name])

    def _parse_point_id(self, id: str) -> Tuple[str, str]:
        id, suffix = id.rsplit("-", 1)
        for collection_name, suffix in ID_SUFFIXES.items():
            if type == suffix:
                return id, collection_name
        raise ValueError(f"Invalid id {id}")
