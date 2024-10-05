from typing import Annotated
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langgraph.prebuilt import InjectedState
from llm.groq_llm import groq_llm_initializer

@tool
def text_response_tool(query: str, state: Annotated[dict, InjectedState]) -> str:
    """
    You are an expert data analyst, you are given a dataset and if the user wants in return response as text, based on this data answer the users query.

    Args:
        query (str): the query to answer based on the dataset
        state (InjectedState): The state object containing the dataset.
    
    Returns:
        str: respone to that query
    """

    text_prompt = PromptTemplate(
    input_variables=[ "query","dataset"],
    template="""
    You are an expert data analyst, you are given a dataset and baesd on this data answer the users query.
    
    dataset: {dataset}

    Follow these guidlines:
    1. Understand the main intent of the query.
    2. Provide a direct answer to the question.
    3. If additional information is needed, ask for clarification.
    4. End the response with an offer for further assistance.

    User Query: {query}

    Your response:
    """
    )

    state["type"] = "text"

    llm = groq_llm_initializer()

    chain = text_prompt | llm

    response = chain.invoke({"query":query,"dataset":state["dataset"]})

    return response.content