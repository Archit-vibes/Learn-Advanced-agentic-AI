from typing import TypedDict
from typing_extensions import Annotated

from langchain_groq import ChatGroq
from langgraph.graph import state , StateGraph , START , END
from langgraph.graph.message import add_messages

import os
from dotenv import load_dotenv

load_dotenv()

