from typing import TypedDict
from typing_extensions import Annotated

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph import state , StateGraph , START , END
from langgraph.graph.message import add_messages

from langchain_tavily import TavilySearch

from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

import os
from dotenv import load_dotenv

load_dotenv()


class State(TypedDict):
    messages:Annotated[list , add_messages]

llm = ChatGroq(model="llama-3.3-70b-versatile")


websearch_tool = TavilySearch(max_results=2)

@tool
def multiply(a:int , b:int)->int:
    """Divide a and b

    Args:
        a : first integer
        b : second integer

    Returns:
        int : output integer
    
    """
    return a/b

tools = [websearch_tool , multiply]

llm_with_tools = llm.bind_tools(tools)

# Node define
def tool_calling_node(state:State):
    return {"messages":[llm_with_tools.invoke(state["messages"])]}


# Graph
graph_builder = StateGraph(State)
graph_builder.add_node("tool_calling_node" , tool_calling_node)
graph_builder.add_node("tools" , ToolNode(tools))


# Edge Define
graph_builder.add_edge(START , "tool_calling_node")
graph_builder.add_conditional_edges("tool_calling_node" , tools_condition)
graph_builder.add_edge("tools" , END)

# Compiler

graph = graph_builder.compile()

response = graph.invoke({"messages":"divide 9 by 3"})

print(response)
print("\n \n")
print(response['messages'][-1].content)