import unittest
from unittest.mock import patch
from db_constants import get_db_scema, execute_read_query
import psycopg2.extras

class TestDBConstants(unittest.TestCase):

    @patch('db_constants.psycopg2.connect')
    def test_get_db_scema(self, mock_connect):
        # Mock the database connection
        mock_conn = mock_connect.return_value
        mock_cursor = mock_conn.cursor.return_value
        # Mock the cursor factory
        mock_cursor_factory = mock_cursor
        # Mock the query execution and result
        mock_cursor.fetchall.return_value = [('table1',), ('table2',)]
        # Call the function
        result = get_db_scema()
        # Assert the result
        expected_result = '[["table1"], ["table2"]]'
        self.assertEqual(result, expected_result)
        # Assert the database connection is closed
        mock_cursor_factory.close.assert_called_once()
        mock_conn.close.assert_called_once()


    @patch('db_constants.psycopg2.connect')
    def test_execute_read_query_success(self, mock_connect):
        # Mock the database connection
        mock_conn = mock_connect.return_value
        mock_cursor = mock_conn.cursor.return_value
        mock_cursor_factory = mock_cursor

        # Mock the query execution and result
        mock_cursor.fetchall.return_value = [('result1',), ('result2',)]

        # Call the function
        success, result = execute_read_query('SELECT * FROM table')

        # Assert the success status and result
        self.assertTrue(success)
        expected_result = '[["result1"], ["result2"]]'
        self.assertEqual(result, expected_result)

        # Assert the database connection is closed
        mock_cursor.close.assert_called_once()
        mock_conn.close.assert_called_once()

    @patch('db_constants.psycopg2.connect')
    def test_execute_read_query_failure(self, mock_connect):
        # Mock the database connection
        mock_conn = mock_connect.return_value
        mock_cursor = mock_conn.cursor.return_value

        # Mock the query execution and raise an exception
        mock_cursor.fetchall.side_effect = Exception('Error executing query')

        # Call the function
        success, error = execute_read_query('SELECT * FROM table')

        # Assert the failure status and error message
        self.assertFalse(success)
        expected_error = 'Error executing query'
        self.assertEqual(error, expected_error)

        # Assert the database connection is closed
        mock_cursor.close.assert_called_once()
        mock_conn.close.assert_called_once()

if __name__ == '__main__':
    unittest.main()