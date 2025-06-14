from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
import os
import logging

from .state import AgentState  # TypedDict
from .tools import tools

# --- Force correct model name ---
os.environ["GROQ_MODEL_NAME"] = "llama3-8b-8192"

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')
logger = logging.getLogger(__name__)

# --- LLM Setup ---
llm = ChatGroq(
    temperature=0,
    model_name="llama3-8b-8192",  # ✅ Force use of correct model
    groq_api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com"  # ✅ Correct base
)

# --- Decision Nodes ---

async def categorize(state: AgentState) -> AgentState:
    logger.info("--- CATEGORIZE NODE ---")
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Categorize this customer support query as one of: billing, technical, general, escalate. Respond with only the category name."),
            ("user", "{input}")
        ])
        chain = prompt | llm
        result = await chain.ainvoke({"input": state["messages"][-1].content})
        category = result.content.strip().lower()
        return {**state, "category": category}
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
        sentiment = result.content.strip().lower()
        return {**state, "sentiment": sentiment}
    except Exception as e:
        logger.error(f"Sentiment error: {e}")
        return {**state, "error": str(e)}

# --- Sub-Agent Handlers ---

async def handle_billing(state: AgentState) -> AgentState:
    logger.info("--- HANDLE BILLING ---")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a billing agent."),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    return await _run_sub_agent(state, prompt, [tools[0]])

async def handle_general(state: AgentState) -> AgentState:
    logger.info("--- HANDLE GENERAL NODE ---")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful customer support assistant."),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    return await _run_sub_agent(state, prompt, tools)

async def handle_technical(state: AgentState) -> AgentState:
    logger.info("--- HANDLE TECHNICAL ---")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a technical support specialist. Provide step-by-step troubleshooting advice."),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    return await _run_sub_agent(state, prompt, tools)

async def handle_escalation(state: AgentState) -> AgentState:
    logger.info("--- HANDLE ESCALATION ---")
    response = AIMessage(content="This issue has been escalated to our human support team.")
    return {**state, "messages": state["messages"] + [response]}

# --- Sub-Agent Executor ---

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
