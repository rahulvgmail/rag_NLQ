import psycopg2
from unittest.mock import MagicMock

from database import db_constants

class DBReader:
    def __init__(self):
        self.conn = psycopg2.connect(
            host=db_constants.DB_HOST,
            port=db_constants.DB_PORT,
            database=db_constants.DB_NAME,
            user=db_constants.DB_USER,
            password=db_constants.DB_PASSWORD
        )

    def execute_query(self, query):
        cursor = self.conn.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        cursor.close()
        return result
