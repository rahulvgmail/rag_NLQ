from typing import Tuple
from dotenv import load_dotenv
import os

import psycopg2
import json
import psycopg2.extras

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')

postgres_user = os.getenv('POSTGRES_USER')
postgres_user_password = os.getenv('POSTGRES_USER_PASSWORD')
postgres_db = os.getenv('POSTGRES_DB')
postgres_port = os.getenv('POSTGRES_PORT')
postgres_host = os.getenv('POSTGRES_HOST')



class pgConnection:
    def __init__(self):
        self.db_url = f'postgresql://{postgres_user}:{postgres_user_password}@{postgres_host}:{postgres_port}/{postgres_db}'
        self.conn = psycopg2.connect(self.db_url)
        self.conn.set_session(readonly=True)
        self.cursor = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    def execute_query(self, query):
        self.cursor.execute(query)
        result = self.cursor.fetchall()
        return result

    def close_connection(self):
        self.cursor.close()
        self.conn.close()
    
    def get_connection(self):
        return self.db_url

    def get_db_scema(self ):
        # Connect to the database
        conn = psycopg2.connect(self.db_url)
        # Create a read-only cursor
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
        # Execute the query to get the schema
        cursor.execute("SELECT * FROM information_schema.columns where table_schema not in ('pg_catalog', 'information_schema') order by table_schema, table_name ")
        # Fetch all the table names
        tables = cursor.fetchall()

#    tables = json.dumps(tables)
    # Close the cursor and connection


#function to execute read queries
def execute_read_query(query) -> Tuple[bool, str]:
    try:
        db_url = f'postgresql://{postgres_user}:{postgres_user_password}@{postgres_host}:{postgres_port}/{postgres_db}'
        # Connect to the database
        conn = psycopg2.connect(db_url)
        # Create a read-only cursor
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        # Execute the query
        cursor.execute(query)
        # Fetch the results
        result = cursor.fetchall()
        # Close the cursor and connection
        cursor.close()
        conn.close()
        # Return the success status and result as a tuple
        return (True, json.dumps(result))
    except Exception as e:
        cursor.close()
        conn.close()
        # Return the error status and error message as a tuple
        return (False, str(e))
