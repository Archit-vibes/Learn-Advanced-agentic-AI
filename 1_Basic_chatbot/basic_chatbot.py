from typing import Annotated
from typing_extensions import TypedDict

from langchain_groq import ChatGroq

import os
from dotenv import load_dotenv

from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages

load_dotenv()


class State(TypedDict):
    messages:Annotated[list , add_messages]

llm = ChatGroq(model='qwen/qwen3-32b')

def chatbot(state:State):
    return {'messages':[llm.invoke(state['messages'])]}

graph_builder = StateGraph(State)

#adding node
graph_builder.add_node("Chatbot_call" , chatbot)
#adding edges
graph_builder.add_edge(START,"Chatbot_call")
graph_builder.add_edge("Chatbot_call" , END)


graph = graph_builder.compile()

res = graph.invoke({'messages':'hi'})

print(res['messages'][-1].content)