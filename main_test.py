
import os
from dotenv import load_dotenv

from vector_embed.openai_embeddings import qdrant_store


if __name__ == "__main__":
    # Initialize OpenAIEmbeddings
    # Load environment variables from .env file
    load_dotenv()

    # Get the API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")

    vector_db_url = os.getenv("QDRANT_URL")

    qdrant_instance = qdrant_store(api_key, vector_db_url)

    # Generate document collection
    documents = ['document1', 'document2', 'document3']
    generated_embeddings = qdrant_instance.generate_embedding(documents)
    #embeddings.generate_embeddings(documents)

    # Save embeddings to output file
    output_file = '/path/to/output/file.txt'
    for doc in documents:
        qdrant_instance.add_documentation(doc)

    # Initialize QdrantEmbeddings
    #QdrantEmbeddings(api_key, database_url, vector_database_url)


    # Fetch data from Qdrant
    question = 'sample question'
    qdrant_instance.add_question_sql("how many cities in USA", "select count(*) from cities where country = \'USA\'")
    string_with_quote = '\''
    sql_results = qdrant_instance.get_similar_question_sql(question)
    ddl_results = qdrant_instance.get_related_ddl(question)
    documentation_results = qdrant_instance.get_related_documentation(question)

    # Print the results
    print("SQL Results:")
    for result in sql_results:
        print(result)

    print("DDL Results:")
    for result in ddl_results:
        print(result)

    print("Documentation Results:")
    for result in documentation_results:
        print(result)

    # Remove data from Qdrant
    qdrant_instance.remove_training_data(question)