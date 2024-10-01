from langgraph.graph import StateGraph, START, END
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from tools.chart_response_tool import chart_response_tool
from tools.text_response_tool import text_response_tool
from llm.groq_llm import groq_llm_initializer

class State(TypedDict):
    messages: Annotated[list, add_messages]
    dataset: str

memory = MemorySaver()

def Initializer():

    tools = [text_response_tool,chart_response_tool]

    llm = groq_llm_initializer()

    llm_with_tools = llm.bind_tools(tools)

    def tool_calling_llm(state: State):
        response = llm_with_tools.invoke(state["messages"])
        
        # Check if the response contains a tool call
        if 'tool_calls' in response and response['tool_calls']:
            # If there's a tool call, return it as is
            return {"messages": [response]}
        else:
            # If there's no tool call, return only the content
            return {"messages": [response]}
        
    
    builder = StateGraph(State)


    # Define nodes: these do the work
    builder.add_node("assistant", tool_calling_llm)
    builder.add_node("tools", ToolNode(tools))
    # Define edges: these determine how the control flow moves
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )
    builder.add_edge("tools", "assistant")

    # The checkpointer lets the graph persist its state
    # this is a complete memory for the entire graph.
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)

    return graph