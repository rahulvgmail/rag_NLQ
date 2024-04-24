import os

from langchain.sql_database import SQLDatabase
from db_constants import get_connection 
#from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain.chains.openai_tools import create_extraction_chain_pydantic
from langchain_core.pydantic_v1 import BaseModel, Field

class Table(BaseModel):
    """Table in SQL database."""

    name: str = Field(description="Name of table in SQL database.")

def question_to_sql(question: str) -> str:
    """
    Convert a question to a SQL query.

    Parameters:
        question (str): The question to convert.

    Returns:
        str: The SQL query.
    """
    db = SQLDatabase.from_uri( get_connection())
    print(db.get_usable_table_names())
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
 

    table_names = "\n".join(db.get_usable_table_names())
    system = f"""Return the names of ALL the SQL tables that MIGHT be relevant to the user question. \
    The tables are:

    {table_names}

    Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed."""
    table_chain = create_extraction_chain_pydantic(Table, llm, system_message=system)
    table_chain.invoke({"input": "What are all the genres of Alanis Morisette songs"})
#    agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)
 #   agent_executor.invoke({"input": question})
    return "hope this works"



def main():
    question = input("Enter a question: ")
    sql_query = question_to_sql(question)
    print("SQL Query:", sql_query)

if __name__ == "__main__":
    main()