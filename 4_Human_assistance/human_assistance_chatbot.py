import os
from dotenv import load_dotenv

from typing import TypedDict
from typing_extensions import Annotated

from langgraph.graph import StateGraph,state , START , END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt , Command
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

from langchain_groq import ChatGroq
from langchain_tavily.tavily_search import TavilySearch
from langchain_core.tools import tool

load_dotenv()

mem = MemorySaver()

class State(TypedDict):
    messages:Annotated[list , add_messages]

#-------LLM initialization

llm = ChatGroq(model="openai/gpt-oss-120b")

#------Tool define

@tool
def human_assistance(query):
    """ If information about the user or their organization is missing, ALWAYS call the human_assistance tool. Do not ask the user directly.
    """
    human_res = interrupt({"query":query})
    return human_res['data']

websearch_tool = TavilySearch(max_results=2)

tools = [human_assistance , websearch_tool]

#----llm with tools

llm_with_tools = llm.bind_tools(tools)

#-----node define

def llm_node(state:State):
    return {"messages":[llm_with_tools.invoke(state['messages'])]}

#----add node

graph_builder = StateGraph(State)

graph_builder.add_node("llm_node" , llm_node)
graph_builder.add_node("tools" , ToolNode(tools))

#----add edges

graph_builder.add_edge(START , "llm_node")
graph_builder.add_conditional_edges("llm_node" , tools_condition)
graph_builder.add_edge("tools" , "llm_node")

agent = graph_builder.compile(checkpointer=mem)

config_mem = {'configurable' : {"thread_id" : 1}}

ask_for_assistance = agent.invoke({"messages":"What do you know about my organization ?"} , config=config_mem)

print(ask_for_assistance["messages"][-1].content)


human_res = ("User's organization is Amazon")

human_command = Command(resume={"data": human_res})

result = agent.invoke(human_command , config=config_mem)

print(result["messages"][-1].content)