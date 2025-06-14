from typing import List, Optional
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph
import logging
import os

from .tools import tools
from .state import AgentState

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')
logger = logging.getLogger(__name__)

# --- ChatGroq Setup (API key hardcoded for now) ---
llm = ChatGroq(
    temperature=0,
    model_name="llama3-8b-8192",
    groq_api_key="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",
    base_url="https://api.groq.com"
)

# --- Decision Nodes ---

async def categorize(state: AgentState) -> AgentState:
    logger.info("--- CATEGORIZE NODE ---")
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Categorize this customer query as one of:
- billing
- technical
- general
- escalate (ONLY if the customer explicitly asks for human support)

Respond with ONLY the category name."""),
            ("user", "{input}")
        ])
        chain = prompt | llm
        result = await chain.ainvoke({"input": state["messages"][-1].content})
        return {**state, "category": result.content.strip().lower()}
    except Exception as e:
        logger.error(f"Categorization error: {e}")
        return {**state, "error": str(e)}

async def analyze_sentiment(state: AgentState) -> AgentState:
    logger.info("--- ANALYZE SENTIMENT NODE ---")
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Analyze sentiment as one of: positive, neutral, negative. Respond with only the sentiment."),
            ("user", "{input}")
        ])
        chain = prompt | llm
        result = await chain.ainvoke({"input": state["messages"][-1].content})
        return {**state, "sentiment": result.content.strip().lower()}
    except Exception as e:
        logger.error(f"Sentiment error: {e}")
        return {**state, "error": str(e)}

async def handle_billing(state: AgentState) -> AgentState:
    logger.info("--- HANDLE BILLING ---")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a billing assistant. Help the customer solve any billing-related issues clearly and politely.
If they are not satisfied, offer to escalate to a human."""),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    return await _run_sub_agent(state, prompt, [tools[0]])

async def handle_general(state: AgentState) -> AgentState:
    logger.info("--- HANDLE GENERAL ---")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful customer support assistant. Resolve the issue clearly and professionally.
If the customer is not satisfied, kindly offer to escalate the issue to a human agent."""),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    return await _run_sub_agent(state, prompt, tools)

async def handle_technical(state: AgentState) -> AgentState:
    logger.info("--- HANDLE TECHNICAL ---")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a technical support assistant. Guide the customer step-by-step to resolve technical issues.
If they're still stuck, politely offer to escalate to a human agent."""),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    return await _run_sub_agent(state, prompt, tools)

async def handle_escalation(state: AgentState) -> AgentState:
    logger.info("--- HANDLE ESCALATION ---")
    response = AIMessage(content="I'm escalating this to our human support team. They'll be with you shortly.")
    return {**state, "messages": state["messages"] + [response]}

# --- Sub-agent Executor ---

async def _run_sub_agent(state: AgentState, prompt, toolset) -> AgentState:
    agent = prompt | llm.bind_tools(toolset)
    try:
        response = await agent.ainvoke({
            "messages": state["messages"],
            "agent_scratchpad": state.get("agent_scratchpad", [])
        })
        updated_msgs = state["messages"] + [AIMessage(content=response.content, tool_calls=response.tool_calls)]
        return {
            **state,
            "messages": updated_msgs,
            "agent_scratchpad": state.get("agent_scratchpad", [])
        }
    except Exception as e:
        logger.error(f"Sub-agent error: {e}")
        return {**state, "error": str(e)}

# --- Routing Logic ---

def get_next_node(state: AgentState) -> str:
    if state.get("error"):
        return "handle_escalation"
    if state.get("category") == "escalate":
        return "handle_escalation"
    elif state.get("category") == "billing":
        return "handle_billing"
    elif state.get("category") == "technical":
        return "handle_technical"
    else:
        return "handle_general"

# --- Graph Definition ---

def create_agent() -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("categorize", categorize)
    workflow.add_node("analyze_sentiment", analyze_sentiment)
    workflow.add_node("handle_billing", handle_billing)
    workflow.add_node("handle_technical", handle_technical)
    workflow.add_node("handle_general", handle_general)
    workflow.add_node("handle_escalation", handle_escalation)

    workflow.set_entry_point("categorize")
    workflow.add_edge("categorize", "analyze_sentiment")
    workflow.add_conditional_edges("analyze_sentiment", get_next_node)

    for node in ["handle_billing", "handle_technical", "handle_general", "handle_escalation"]:
        workflow.set_finish_point(node)

    return workflow.compile()

# --- Exported App ---
app = create_agent()
