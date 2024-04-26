
import re

from dotenv import load_dotenv
from vector_embed.openai_embeddings import qdrant_store
import openai
import os

load_dotenv()

class openai_agent:
    def __init__(self, qdrant_store: qdrant_store):
        
        self._embedding_store = qdrant_store
        self._temperature = 0.7
        self._max_tokens = 500
        self._client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._model = "gpt-3.5-turbo"

    def act(self, state):
        return self.model.predict(state)

    def train(self, state, action, reward, next_state, done):
        self.model.train(state, action, reward, next_state, done)

    def save(self):
        self.model.save()

    def load(self):
        self.model.load()
    def str_to_approx_token_count(self, string: str) -> int:
        return len(string) / 4

# real code

    def log(self, message: str):
        print(message)
# copied from vanna 

    def generate_sql(self, question: str, **kwargs) -> str:
        """

        Args:
            question (str): The question to generate a SQL query for.

        Returns:
            str: The SQL query that answers the question.
        """
        initial_prompt = None
        question_sql_list = self._embedding_store.get_similar_question_sql(question, **kwargs)
        ddl_list = self._embedding_store.get_related_ddl(question, **kwargs)
        doc_list = self._embedding_store.get_related_documentation(question, **kwargs)
        prompt = self.get_sql_prompt(
            initial_prompt=initial_prompt,
            question=question,
            question_sql_list=question_sql_list,
            ddl_list=ddl_list,
            doc_list=doc_list,
            **kwargs,
        )
        self.log(prompt)
        llm_response = self.submit_prompt(prompt, **kwargs)
        self.log(llm_response.choices[0].message.content)
        return self.extract_sql(llm_response.choices[0].message.content)

    def extract_sql(self, llm_response: str) -> str:
        # If the llm_response is not markdown formatted, extract sql by finding select and ; in the response
        sql = re.search(r"SELECT.*?;", llm_response, re.DOTALL)
        if sql:
            self.log(f"Output from LLM: {llm_response} \nExtracted SQL: {sql.group(0)}"
            )
            return sql.group(0)

        # If the llm_response contains a CTE (with clause), extract the sql bewteen WITH and ;
        sql = re.search(r"WITH.*?;", llm_response, re.DOTALL)
        if sql:
            self.log(f"Output from LLM: {llm_response} \nExtracted SQL: {sql.group(0)}")
            return sql.group(0)
        # If the llm_response contains a markdown code block, with or without the sql tag, extract the sql from it
        sql = re.search(r"```sql\n(.*)```", llm_response, re.DOTALL)
        if sql:
            self.log(f"Output from LLM: {llm_response} \nExtracted SQL: {sql.group(1)}")
            return sql.group(1)

        sql = re.search(r"```(.*)```", llm_response, re.DOTALL)
        if sql:
            self.log(f"Output from LLM: {llm_response} \nExtracted SQL: {sql.group(1)}")
            return sql.group(1)

        return llm_response
    
    def submit_prompt(self, prompt: str) -> str:
        """
        Submit the prompt to the LLM and return the response.

        Args:
            prompt (str): The prompt to submit to the LLM.

        Returns:
            str: The response from the LLM.
        """
        response = self._client.chat.completions.create(
            model=self._model, 
            messages=prompt,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        return response
    def add_documentation_to_prompt(
        self,
        initial_prompt: str,
        documentation_list: list[str],
        max_tokens: int = 14000,
    ) -> str:
        if len(documentation_list) > 0:
            initial_prompt += "\nYou may use the following documentation as a reference for what tables might be available. Use responses to past questions also to guide you:\n\n"

            for documentation in documentation_list:
                if (
                    self.str_to_approx_token_count(initial_prompt)
                    + self.str_to_approx_token_count(documentation)
                    < max_tokens
                ):
                    initial_prompt += f"{documentation}\n\n"

        return initial_prompt

    def add_sql_to_prompt(
        self, initial_prompt: str, sql_list: list[str], max_tokens: int = 14000
    ) -> str:
        if len(sql_list) > 0:
            initial_prompt += "\nYou may use the following SQL statements as a reference for what tables might be available. Use responses to past questions also to guide you:\n\n"

            for question in sql_list:
                if (
                    self.str_to_approx_token_count(initial_prompt)
                    + self.str_to_approx_token_count(question["sql"])
                    < max_tokens
                ):
                    initial_prompt += f"{question['question']}\n{question['sql']}\n\n"

        return initial_prompt


    def get_sql_prompt(
        self,
        question: str,
        doc_list: list,
        question_sql_list: list,
        **kwargs,
    ):
        """
        Example:
        ```python
        vn.get_sql_prompt(
            question="What are the top 10 customers by sales?",
            question_sql_list=[{"question": "What are the top 10 customers by sales?", "sql": "SELECT * FROM customers ORDER BY sales DESC LIMIT 10"}],
            ddl_list=["CREATE TABLE customers (id INT, name TEXT, sales DECIMAL)"],
            doc_list=["The customers table contains information about customers and their sales."],
        )

        ```

        This method is used to generate a prompt for the LLM to generate SQL.

        Args:
            question (str): The question to generate SQL for.
            question_sql_list (list): A list of questions and their corresponding SQL statements.
            ddl_list (list): A list of DDL statements.
            doc_list (list): A list of documentation.

        Returns:
            any: The prompt for the LLM to generate SQL.
        """

        initial_prompt = "This is a Postgres database. The user provides a question and you provide SQL. You will only respond with SQL code and not with any explanations.\n\nRespond with only SQL code. Do not answer with any explanations -- just the code.\n"
        #initial_prompt += "\n\n the table name in the output should be name of the table_schema and table_name joined by a \'.\'. do not include table name in selected columns. the format is only for providing fully qualified table name\n\n"
        initial_prompt += "\n\n please provide fully qualified table name in the output. \n\n"
        initial_prompt += "\n\n instrad of 'select column1, column2 from table_name' please output \n\n"
        initial_prompt += "\n\n  'select column1, column2 from table_schema.table_name' \n\n"
        initial_prompt = self.add_documentation_to_prompt(
            initial_prompt, doc_list, max_tokens=14000
        )
        message_log = [self.system_message(initial_prompt)]
        
        for example in question_sql_list:
            if example is None:
                print("example is None")
            else:
                if example is not None and "question" in example and "sql" in example:
                    message_log.append(self.user_message(example["question"]))
                    message_log.append(self.assistant_message(example["sql"]))

        message_log.append(self.user_message(question))

        return message_log
    
    def system_message(self, message: str) -> any:
        return {"role": "system", "content": message}

    def user_message(self, message: str) -> any:
        return {"role": "user", "content": message}

    def assistant_message(self, message: str) -> any:
        return {"role": "assistant", "content": message}
