from typing import Annotated
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from llm.groq_llm import groq_llm_initializer
from langgraph.prebuilt import InjectedState
import re

@tool(return_direct=True)
def table_response_tool(query: str, state: Annotated[dict, InjectedState]) -> str:
    """
    use this when the user asks for a table in return.


    Args:
        query (str): The user's query to answer.
        dataset (str): data 
    Returns:
        str: return json formatted table
    """
    
    # Access the dataset from the state
    dataset = state["dataset"]

    table_prompt = PromptTemplate(
    input_variables=["query", "dataset"],
    template="""
    You're a data Analyst expert and create a table based on the needs of the quesy of the user.
    
    dataset: {dataset}

    Follow these guidlines:
    1. generate a super detailed table in a json formatted table based on the query of the user and the data provided.
    2. if there are any data that is needed to be calculated, do the necessessary calculation for them.
    3. return meaningful titles for every column.
    
    User Query: {query}
    """
    )

    state["type"] = "table"

    llm = groq_llm_initializer()

    chain = table_prompt | llm

    response = chain.invoke({"query": query, "dataset": dataset})

    print(response.content)

    # Regular expression to match the content between ```code``` blocks
    pattern = r'```json(.*?)```'

    match = re.search(pattern, response.content, re.DOTALL)
    
    print(match.group(1).strip())

    if match:
        # Return the extracted code (strip to remove leading/trailing whitespaces)
        return str(match.group(1).strip())
    
    return response.content