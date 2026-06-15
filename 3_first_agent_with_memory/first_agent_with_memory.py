from dotenv import load_dotenv
import os

from typing import TypedDict
from typing_extensions import Annotated

from langgraph.graph import state,StateGraph,START,END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

from langchain_core.tools import tool
from langchain_tavily.tavily_search import TavilySearch
from langchain_groq import ChatGroq

load_dotenv()

mem = MemorySaver()

#--make class

class State(TypedDict):
    messages:Annotated[list , add_messages]

#--model config

llm = ChatGroq(model="llama-3.3-70b-versatile")

#--tool defining

@tool
def divide(a:int , b:int)->int:
    """Divide a by b

    Args:
        a : first integer
        b : second integer

    Returns:
        int : output integer
    
    """
    return a/b

websearch_tool = TavilySearch(max_results=2)

tools = [websearch_tool , divide]

#--config model with tools

llm_with_tools = llm.bind_tools(tools)

#--Starts graph

graph_builder = StateGraph(State)

#--create node

def llm_node(state:State):
    return {'messages':[llm_with_tools.invoke(state['messages'])]}

#--add node

graph_builder.add_node("llm_node" , llm_node)
graph_builder.add_node("tools" , ToolNode(tools))

#--make edges

graph_builder.add_edge(START , "llm_node")
graph_builder.add_conditional_edges("llm_node" , tools_condition)
graph_builder.add_edge("tools" , "llm_node")

agent = graph_builder.compile(checkpointer=mem)

config_mem = {'configurable':{'thread_id':"1"}}

res_ans = agent.invoke({'messages':"Hey my name is archit"} , config=config_mem)

print(res_ans['messages'][-1].content)

res_qsn = agent.invoke({"messages":"Hey , what is my name ?"} , config=config_mem)

print("\n \n")

print(res_qsn['messages'][-1].content)

