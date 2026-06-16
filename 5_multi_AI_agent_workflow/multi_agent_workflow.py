from dotenv import load_dotenv

from langgraph.graph import MessagesState , START , END , StateGraph


from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage , AIMessage

from pydantic import BaseModel

from typing import Literal , Annotated

load_dotenv()

#-----------------------------------LLM initalization

llm = ChatGroq(model="llama-3.3-70b-versatile")

#-----------------------------------Supervisor/Orchestrator

 #--------Supervisor State

class SupervisorState(MessagesState):
    """Supervisor Class for LLM"""
    next_agent:str
    research:str
    analysis:str
    final_report:str
    task_complete:bool = False
    current_task:str

#---------Supervisor Prompt Function

 #---Supervisor Literal Class

class supervisor_schema(BaseModel):
    next_agent_name:Literal["researcher" , "analyst" , "writer" , "DONE"]

#---Supervisor Chain

def Supervisor_chain():

    structured_llm = llm.with_structured_output(supervisor_schema)

    supervisor_prompt = PromptTemplate(
        input_variables=["task" , "has_research" , "has_analysis" , "has_report" ] ,
        
        template="""
                You are a supervisor managing a team of agents:

                1. Researcher - Gathers information and data
                2. Analyst - Analyzes data and provides insights
                3. Writer - Creates reports and summaries

                Based on the current state and conversation, decide which agent should work next.
                If the task is complete, respond with 'DONE'.

                Current state:
                - Has research data: {has_research}
                - Has analysis: {has_analysis}
                - Has report: {has_report}

                Task:
                {task}

                Respond with ONLY the agent name (researcher/analyst/writer) or 'DONE'.
                """)
    
    return supervisor_prompt | structured_llm

#-----SUpervisor Agent

def Supervisor_agent(state:SupervisorState):
    """Supervisor agent which orchrestratos sub-agents"""
    
    task = state["current_task"]


    #---Check whats done

    has_research = state.get("research" , "")
    has_analysis = state.get("analysis" , "")
    has_report = state.get("final_report" , "")

    #---Decide which agent use next
    chain = Supervisor_chain()
    decision = chain.invoke({
    "task": task,
    "has_research": bool(state.get("research", "")),
    "has_analysis": bool(state.get("analysis", "")),
    "has_report": bool(state.get("final_report", "")),
})
    decision_text = decision.next_agent_name.lower()

    print("Decision Text : " , decision_text)

    # Determine the next agent

    if decision_text == "done" or has_report:
        next_agent = "end"
        supervisor_msg = "✅ Supervisor: All tasks complete! Great work team."

    elif decision_text == "researcher" or not has_research:
        next_agent = "researcher"
        supervisor_msg = (
            "📋 Supervisor: Let's start with research. "
            "Assigning to Researcher..."
        )

    elif decision_text == "analyst" or (has_research and not has_analysis):
        next_agent = "analyst"
        supervisor_msg = (
            "📋 Supervisor: Research done. "
            "Time for analysis. Assigning to Analyst..."
        )

    elif decision_text == "writer" or (has_analysis and not has_report):
        next_agent = "writer"
        supervisor_msg = (
            "📋 Supervisor: Analysis complete. "
            "Let's create the report. Assigning to Writer..."
        )

    else:
        next_agent = "end"
        supervisor_msg = "✅ Supervisor: Task seems complete."

    return {
        "messages": [AIMessage(content=supervisor_msg)],
        "next_agent": next_agent,
        "current_task": task
    }

#--------------------------------------Researcher Agent

def researcher_agent(state: SupervisorState):
    """Uses LLM to research on a topic"""

    task = state.get("current_task")
    
    # Create research prompt
    research_prompt = f"""
        As a research specialist, provide comprehensive information about: {task}

        Include:
        1. Key facts and background
        2. Current trends or developments
        3. Important statistics or data points
        4. Notable examples or case studies
        
        Be concise but thorough."""
    
    research_res = llm.invoke(research_prompt)
    research_data = research_res.content

    # Create agent message
    agent_message = f"🔍 Researcher: I've completed the research on '{task}'.\n\nKey findings:\n{research_data[:500]}..."

    print("\n \n" , agent_message , "\n \n")

    return {
        "messages": [AIMessage(content=agent_message)],
        "research": research_data,
        "next_agent": "supervisor"
    }

#--------------------------------------Analysis Agent


def analysis_agent(state: SupervisorState):
    """Uses LLM to analyse the research"""

    research = state.get("research", "")
    task = state.get("current_task")
    
    # Create analysis prompt
    analysis_prompt = f"""
        As a data analyst, analyze this research data and provide insights:

        Research Data:
        {research}

        Provide:
        1. Key insights and patterns
        2. Strategic implications
        3. Risks and opportunities
        4. Recommendations

        Focus on actionable insights related to: {task}"""
    
    analysis_res = llm.invoke(analysis_prompt)
    analysis_data = analysis_res.content

    # Create agent message
    agent_message = f"📊 Analyst: I've completed the analysis.\n\nTop insights:\n{analysis_data[:400]}..."

    print("\n \n" , agent_message , "\n \n")

    return {
        "messages": [AIMessage(content=agent_message)],
        "analysis": analysis_data,
        "next_agent": "supervisor"
    }

def writer_agent(state: SupervisorState):
    """Uses LLM to create the final report"""

    task = state.get("current_task")
    research = state.get("research", "")
    analysis = state.get("analysis", "")

    # Create report prompt
    report_prompt = f"""
        As a professional report writer, create a comprehensive final report.

        Task:
        {task}

        Research:
        {research}

        Analysis:
        {analysis}

        Create a well-structured report with:

        1. Executive Summary
        2. Background
        3. Key Findings
        4. Analysis & Insights
        5. Recommendations
        6. Conclusion

        Make the report clear, professional, and actionable.
    """

    report_res = llm.invoke(report_prompt)
    final_report = report_res.content

    # Create agent message
    agent_message = (
        f"📝 Writer: I've completed the final report.\n\n"
        f"Report Preview:\n{final_report[:400]}..."
    )

    print("\n \n" , agent_message , "\n \n")

    return {
        "messages": [AIMessage(content=agent_message)],
        "final_report": final_report,
        "next_agent": "supervisor"
    }

#------------Router Func

def router(state:SupervisorState):
    
    next_agent = state.get("next_agent" , "supervisor")

    if next_agent == "end" or state.get("task_complete", False):
        return END
        
    if next_agent in ["supervisor", "researcher", "analyst", "writer"]:
        return next_agent
        
    return "supervisor"

#--------------workflow

workflow = StateGraph(SupervisorState)

# Add nodes

workflow.add_node("supervisor" , Supervisor_agent)
workflow.add_node("researcher" , researcher_agent)
workflow.add_node("analyst", analysis_agent)
workflow.add_node("writer" , writer_agent)

# Add edges

workflow.add_edge(START, "supervisor")

for node in ["supervisor", "researcher", "analyst", "writer"]:
    workflow.add_conditional_edges(
        node,
        router,
        {
            "supervisor": "supervisor",
            "researcher": "researcher",
            "analyst": "analyst",
            "writer": "writer",
            END: END
        }
    )

graph=workflow.compile()

res = graph.invoke({
    "current_task": "Updates about Tesla"
})

