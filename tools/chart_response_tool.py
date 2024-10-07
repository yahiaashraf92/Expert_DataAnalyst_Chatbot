from typing import Annotated
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from llm.groq_llm import groq_llm_initializer
from langgraph.prebuilt import InjectedState
import re

@tool(return_direct=True)
def chart_response_tool(query: str, state: Annotated[dict, InjectedState]) -> str:
    """
    use this when the user asks for a data visualization, chart or a graph.


    Args:
        query (str): The user's query to answer.
        dataset (str): data 
    Returns:
        str: return suitable chart for the query
    """
    
    # Access the dataset from the state
    dataset = state["dataset"]

    chart_prompt = PromptTemplate(
    input_variables=["query", "dataset"],
    template="""
    You're a data visualization expert and you use your favourite graphing library chart.js for angular only.
    
    dataset: {dataset}

    Follow these guidlines:
    1. Add all the necessary details for the code to be run without any further errors.
    2. Identify the type of data the user is asking about.
    3. Determine the most appropriate chart type (bar, line, pie, etc.).
    4. Specify the data points and labels needed for the chart, add the data in the chart config manually don't use variable list.
    5. return the output Chart Configurations as a json, either it is one chart or many return them as array (WHAT I MEAN BY THE CONTENT OF THE CHART IS THIS : Chart(ctx,CONTENT)).
    6. don't create any variables, add all the data needed for the chart in the chart configurations. 
    7. DO NOT include any preamble text. Do not include explanations or prose, respond only with the generated artifact.   
    8. Make the chart or dashboard color theme more impressive.
    9. Follow the user's indications when creating the graph
    
    User Query: {query}
    """
    )

    state["type"] = "chart"

    llm = groq_llm_initializer()

    chain = chart_prompt | llm

    response = chain.invoke({"query": query, "dataset": dataset})

    print(response.content)

    # Regular expression to match the content between ```code``` blocks
    pattern = r'```typescript(.*?)```'

    match = re.search(pattern, response.content, re.DOTALL)
    
    print(match.group(1).strip())

    if match:
        # Return the extracted code (strip to remove leading/trailing whitespaces)
        return str(match.group(1).strip())
    
    return response.content