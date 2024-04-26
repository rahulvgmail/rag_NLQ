from typing import Union
import psycopg2
from unittest.mock import MagicMock
import pandas as pd
from database import db_constants
from tools.utils import TrainingPlan, TrainingPlanItem
#import tabulate

class DBReader:
    def __init__(self):
        self.db_url = f'postgresql://{db_constants.postgres_user}:{db_constants.postgres_user_password}@{db_constants.postgres_host}:{db_constants.postgres_port}/{db_constants.postgres_db}'
        self._conn = psycopg2.connect(self.db_url)
        self._conn.set_session(readonly=True)
        #self._cursor = self._conn.cursor()

    def execute_query(self, query):
        self._cursor.execute(query)
        result = self.cursor.fetchall()
        return result

    def close_connection(self):
        self.cursor.close()
        self.conn.close()
    
    def get_connection(self):
        return self.db_url

    def run_sql_postgres(self, sql: str) -> Union[pd.DataFrame, None]:
        if self._conn:
            try:
                cs = self._conn.cursor()
                cs.execute(sql)
                results = cs.fetchall()

                # Create a pandas dataframe from the results
                df = pd.DataFrame(
                        results, columns=[desc[0] for desc in cs.description]
                )
                return df

            except psycopg2.Error as e:
                error_message = str(e)
                raise Exception(error_message)
            except Exception as e:
                raise e
    def get_db_scema(self ):
        # Connect to the database
        conn = psycopg2.connect(self.db_url)
        # Create a read-only cursor
        cursor = conn.cursor()
    
        # Execute the query to get the schema
        query = "SELECT * FROM information_schema.columns where table_schema not in ('pg_catalog', 'information_schema') order by table_schema, table_name " 
        res_df = self.run_sql_postgres(query)
        return res_df

    def execute_query(self, query):
        cursor = self.conn.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        cursor.close()
        return result
    def get_training_plan(self, df) -> TrainingPlan:
        """
        This method is used to generate a training plan from an information schema dataframe.

        Basically what it does is breaks up INFORMATION_SCHEMA.COLUMNS into groups of table/column descriptions that can be used to pass to the LLM.

        Args:
            df (pd.DataFrame): The dataframe to generate the training plan from.

        Returns:
            TrainingPlan: The training plan.
        """
        # For each of the following, we look at the df columns to see if there's a match:
        database_column = df.columns[
            df.columns.str.lower().str.contains("database")
            | df.columns.str.lower().str.contains("table_catalog")
        ].to_list()[0]
        schema_column = df.columns[
            df.columns.str.lower().str.contains("table_schema")
        ].to_list()[0]
        table_column = df.columns[
            df.columns.str.lower().str.contains("table_name")
        ].to_list()[0]
        columns = [database_column,
                    schema_column,
                    table_column]
        candidates = ["column_name",
                      "data_type",
                      "comment"]
        matches = df.columns.str.lower().str.contains("|".join(candidates), regex=True)
        columns += df.columns[matches].to_list()

        plan = TrainingPlan([])

        for database in df[database_column].unique().tolist():
            for schema in (
                df.query(f'{database_column} == "{database}"')[schema_column]
                .unique()
                .tolist()
            ):
                for table in (
                    df.query(
                        f'{database_column} == "{database}" and {schema_column} == "{schema}"'
                    )[table_column]
                    .unique()
                    .tolist()
                ):
                    df_columns_filtered_to_table = df.query(
                        f'{database_column} == "{database}" and {schema_column} == "{schema}" and {table_column} == "{table}"'
                    )
                    filtered_df = df_columns_filtered_to_table[columns]
                    doc = f"The following columns are in the {table} table in the {database} database:\n\n"
                    doc += filtered_df.to_markdown()

                    plan._plan.append(
                        TrainingPlanItem(
                            item_type=TrainingPlanItem.ITEM_TYPE_IS,
                            item_group=f"{database}.{schema}",
                            item_name=table,
                            item_value=doc,
                        )
                    )

        return plan





