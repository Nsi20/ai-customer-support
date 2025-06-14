import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

# Define the root directory for your project as the current directory
# This means the script should be run FROM INSIDE your 'ai-customer-support' folder
PROJECT_ROOT = Path(".")

# Define the list of files and directories to create
list_of_files = [
    # Top-level directories
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "config",
    PROJECT_ROOT / "data",
    PROJECT_ROOT / "tests",

    # __init__.py files for Python packages
    PROJECT_ROOT / "src" / "__init__.py",
    PROJECT_ROOT / "config" / "__init__.py",
    PROJECT_ROOT / "data" / "__init__.py",
    PROJECT_ROOT / "tests" / "__init__.py",

    # Core application files in src/
    PROJECT_ROOT / "src" / "main.py", # Included, but will be skipped if exists
    PROJECT_ROOT / "src" / "agent.py",
    PROJECT_ROOT / "src" / "nodes.py",
    PROJECT_ROOT / "src" / "tools.py",
    PROJECT_ROOT / "src" / "state.py",

    # Root project files (will be skipped if they already exist from uv init)
    PROJECT_ROOT / ".env",
    PROJECT_ROOT / ".gitignore",
    PROJECT_ROOT / "pyproject.toml", # Included, but will be skipped if exists
    PROJECT_ROOT / "README.md",      # Included, but will be skipped if exists

    # Test files
    PROJECT_ROOT / "tests" / "test_agent.py",
]

# Dictionary to hold content for specific files
file_contents = {
    PROJECT_ROOT / "src" / "main.py": """# Main entry point for the customer support agent

# from src.agent import app
# from langchain_core.messages import HumanMessage
# import asyncio
# from dotenv import load_dotenv
# import os

# load_dotenv() # Load environment variables from .env file

# async def main():
#     print("Starting AI Customer Support Agent...")
#     print("Type 'exit' or 'quit' to end the conversation.")
#     while True:
#         user_input = input("You: ")
#         if user_input.lower() in ["exit", "quit"]:
#             print("Agent: Goodbye!")
#             break
#         
#         # Ensure your agent 'app' is compiled and ready to use
#         # inputs = {"messages": [HumanMessage(content=user_input)]}
#         # async for s in app.astream(inputs):
#         #     if "__end__" not in s:
#         #         for key, value in s.items():
#         #             # This part might need adjustment based on how you want to display agent steps
#         #             # For simple final output, you might just want to print the last message
#         #             if isinstance(value, dict) and "messages" in value and value["messages"]:
#         #                 last_msg = value["messages"][-1]
#         #                 if last_msg.type == "ai":
#         #                     print(f"Agent: {last_msg.content}")
#         #                     break # Assuming we want to show the final AI response per step
#         #     else:
#         #         print("--- Conversation Flow Ended ---")
#         #
#         # # Alternatively, for a single final response
#         # final_state = await app.ainvoke(inputs)
#         # if final_state["messages"]:
#         #     print(f"Agent: {final_state['messages'][-1].content}")
#         # else:
#         #     print("Agent: I'm sorry, I couldn't process that request.")

# if __name__ == "__main__":
#     asyncio.run(main())
""",
    PROJECT_ROOT / "src" / "agent.py": """# LangGraph agent definition and workflow

# from langgraph.graph import StateGraph, END
# from langchain_groq import ChatGroq # Using ChatGroq for Llama3
# from src.state import AgentState
# from src.nodes import categorize, analyze_sentiment, handle_billing, handle_general, handle_technical, escalate
# from dotenv import load_dotenv
# import os

# load_dotenv() # Load environment variables from .env file

# # Initialize LLM (ensure GROQ_API_KEY is in your .env file)
# # Using Llama3-70b-8192 as requested
# llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")

# # Define the LangGraph workflow
# workflow = StateGraph(AgentState)

# # Add nodes
# workflow.add_node("categorize", categorize)
# workflow.add_node("analyze_sentiment", analyze_sentiment)
# workflow.add_node("handle_billing", handle_billing)
# workflow.add_node("handle_general", handle_general)
# workflow.add_node("handle_technical", handle_technical)
# workflow.add_node("escalate", escalate)

# # Set the entry point
# workflow.set_entry_point("categorize")

# # Define conditional edges
# def route_query(state: AgentState):\n    category = state.get("category")\n    sentiment = state.get("sentiment")\n\n    if sentiment == "angry" or category == "escalate":\n        return "escalate"\n    elif category == "billing":\n        return "handle_billing"\n    elif category == "technical":\n        return "handle_technical"\n    elif category == "general":\n        return "handle_general"\n    else:\n        return "handle_general" # Fallback\n\nworkflow.add_conditional_edges(\n    "analyze_sentiment",\n    route_query,\n    {\n        "escalate": "escalate",\n        "handle_billing": "handle_billing",\n        "handle_general": "handle_general",\n        "handle_technical": "handle_technical",\n    }\n)\n\n# All handling nodes lead to END for now\nworkflow.add_edge("handle_billing", END)\nworkflow.add_edge("handle_general", END)\nworkflow.add_edge("handle_technical", END)\nworkflow.add_edge("escalate", END)\n\n# Compile the graph\napp = workflow.compile()
""",
    PROJECT_ROOT / "src" / "nodes.py": """# Functions for each node in the LangGraph workflow
from typing import List, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph.message import add_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq # Using ChatGroq for Llama3
from langgraph.prebuilt.tool_executor import ToolExecutor
from src.state import AgentState
from src.tools import tools # Import all tools here
from dotenv import load_dotenv
import os

load_dotenv() # Load environment variables

# Initialize LLM (ensure GROQ_API_KEY is in your .env file)
llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")
tool_executor = ToolExecutor(tools)

# Node 1: Categorize
async def categorize(state: AgentState) -> AgentState:
    logging.info("--- CATEGORIZE NODE ---")
    messages = state["messages"]
    latest_message_content = messages[-1].content if messages and messages[-1].content else ""

    # Prompt for categorization
    category_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert at categorizing customer support queries. Categorize the following query into one of 'billing', 'technical', 'general', or 'escalate'. If the query seems complex, urgent, or beyond typical automated handling, categorize it as 'escalate'. Respond with only the category word, lowercase."),
        ("user", "{query}")
    ])
    category_chain = category_prompt | llm

    try:
        response = await category_chain.ainvoke({"query": latest_message_content})
        category = response.content.strip().lower()
        if category not in ["billing", "technical", "general", "escalate"]:
            category = "general" # Default to general if LLM gives an unexpected category
    except Exception as e:
        logging.error(f"Error during categorization: {e}")
        category = "general" # Fallback on error

    logging.info(f"Query categorized as: {category}")
    # Add a system message or an internal AI message for categorization result,
    # without repeating the user's message.
    return {"category": category, "messages": state["messages"]}

# Node 2: Analyze Sentiment
async def analyze_sentiment(state: AgentState) -> AgentState:
    logging.info("--- ANALYZE SENTIMENT NODE ---")
    messages = state["messages"]
    # Get content of the last *user* message for sentiment analysis
    latest_user_message_content = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            latest_user_message_content = msg.content
            break

    sentiment_prompt = ChatPromptTemplate.from_messages([
        ("system", "Analyze the sentiment of the following customer message. Respond with only one word, lowercase: 'positive', 'neutral', 'negative', or 'angry'."),
        ("user", "{message}")
    ])
    sentiment_chain = sentiment_prompt | llm

    try:
        response = await sentiment_chain.ainvoke({"message": latest_user_message_content})
        sentiment = response.content.strip().lower()
        if sentiment not in ["positive", "neutral", "negative", "angry"]:
            sentiment = "neutral" # Default if unexpected sentiment
    except Exception as e:
        logging.error(f"Error during sentiment analysis: {e}")
        sentiment = "neutral" # Fallback on error

    logging.info(f"Sentiment analyzed as: {sentiment}")
    return {"sentiment": sentiment, "messages": state["messages"]}

# Helper function to run a sub-agent (for handle_X nodes)
async def _run_sub_agent(state: AgentState, prompt_template: ChatPromptTemplate, tools_list: List) -> AgentState:
    llm_with_tools = llm.bind_tools(tools_list)
    agent_runnable = prompt_template | llm_with_tools

    response = await agent_runnable.ainvoke({"messages": state["messages"]})

    if response.tool_calls:
        tool_call = response.tool_calls[0] # Assuming one tool call for simplicity
        logging.info(f"Calling tool: {tool_call.name} with args: {tool_call.args}")
        try:
            tool_output = await tool_executor.ainvoke({"tool_name": tool_call.name, "arguments": tool_call.args})
            logging.info(f"Tool output: {tool_output}")
            # Add AI message (tool call) and then ToolMessage (output)
            return {"messages": state["messages"] + [AIMessage(content="", tool_calls=response.tool_calls), ToolMessage(content=str(tool_output), tool_call_id=tool_call.id)]}
        except Exception as e:
            logging.error(f"Error executing tool {tool_call.name}: {e}")
            # If tool fails, return a message indicating failure and revert to general or escalate
            return {"messages": state["messages"] + [AIMessage(content=f"I encountered an error while trying to perform an action related to your request. Please try again or provide more details.", tool_calls=response.tool_calls)], "category": "general"}
    else:
        # If LLM generated a direct response without calling a tool
        return {"messages": state["messages"] + [AIMessage(content=response.content)]}

# Node 3: Handle Billing
async def handle_billing(state: AgentState) -> AgentState:
    logging.info("--- HANDLE BILLING NODE ---")
    billing_llm_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a billing support agent. Respond to the customer's billing query. Use the 'process_billing_query' tool if you need to fetch specific invoice details or explain charges based on a given query. If the query is unclear or requires human review, suggest escalation."),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    return await _run_sub_agent(state, billing_llm_prompt, [tools[0]]) # tools[0] should be process_billing_query

# Node 4: Handle General
async def handle_general(state: AgentState) -> AgentState:
    logging.info("--- HANDLE GENERAL NODE ---")
    general_llm_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a general customer support agent. Answer the customer's general query. Use the 'provide_general_info' tool for common FAQs or general information. If you cannot answer, suggest providing more details or escalating."),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    return await _run_sub_agent(state, general_llm_prompt, [tools[1]]) # tools[1] should be provide_general_info

# Node 5: Handle Technical
async def handle_technical(state: AgentState) -> AgentState:
    logging.info("--- HANDLE TECHNICAL NODE ---")
    technical_llm_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a technical support agent. Help the customer troubleshoot their technical issue. Use the 'troubleshoot_technical_issue' tool to guide them through steps or check system status. If the issue is complex, persistent, or requires live support, suggest escalation."),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    return await _run_sub_agent(state, technical_llm_prompt, [tools[2]]) # tools[2] should be troubleshoot_technical_issue

# Node 6: Escalation Node
async def escalate(state: AgentState) -> AgentState:
    logging.info("--- ESCALATE NODE ---")
    # Extract the original user query for the escalation reason
    original_query = ""
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            original_query = msg.content
            break

    reason = f"Customer query: '{original_query}'. Sentiment: {state.get('sentiment', 'N/A')}. Category: {state.get('category', 'N/A')}. Requires human intervention."
    
    # Call the escalate_to_human tool
    escalation_response = await tool_executor.ainvoke({"tool_name": "escalate_to_human", "arguments": {"reason": reason}})
    
    # Add a final message indicating escalation
    final_message = AIMessage(content=f"I'm sorry, I need to escalate your query to a human agent. {escalation_response} Please provide your contact details and a human will be in touch shortly.")
    return {"messages": state["messages"] + [final_message]}
""",
    PROJECT_ROOT / "src" / "tools.py": """# Definitions of LangChain tools for the agent
from langchain_core.tools import tool
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

@tool
def process_billing_query(query: str, customer_id: str = "Unknown") -> str:
    \"\"\"Processes a billing-related query, e.g., fetching invoice, explaining charges.
    Args:
        query (str): The specific billing question or task (e.g., "get my last invoice", "explain data charges").
        customer_id (str, optional): The customer's unique ID. Defaults to "Unknown".
    \"\"\"
    logging.info(f"Tool: Processing billing query for customer {customer_id}: {query}")
    # --- MOCK IMPLEMENTATION ---
    if "invoice" in query.lower():
        return "I've fetched your last invoice. It shows a total of $55.00 for service period Oct-Nov 2024."
    elif "charges" in query.lower():
        return "The recent charges are for premium support and extended warranty. Would you like a breakdown?"
    else:
        return "I can help with billing queries. Please specify if you need invoice details, charge explanations, or payment history."

@tool
def provide_general_info(query: str) -> str:
    \"\"\"Provides general information or answers common FAQs.
    Args:
        query (str): The general information query (e.g., "what are your hours", "where is your office").
    \"\"\"
    logging.info(f"Tool: Providing general info for query: {query}")
    # --- MOCK IMPLEMENTATION ---
    if "hours" in query.lower() or "operating" in query.lower():
        return "Our operating hours are Monday to Friday, 9 AM to 5 PM local time."
    elif "contact" in query.lower() or "phone" in query.lower():
        return "You can contact us at 1-800-555-1234 or visit our 'Contact Us' page on the website."
    elif "return policy" in query.lower():
        return "Our return policy allows returns within 30 days of purchase, provided the item is in original condition with packaging."
    else:
        return "I can provide general information. Please ask a specific question like 'What are your hours?' or 'What is your return policy?'"

@tool
def troubleshoot_technical_issue(query: str, device_type: str = "general") -> str:
    \"\"\"Helps troubleshoot technical problems, e.g., connectivity issues, software errors.
    Args:
        query (str): The description of the technical issue.
        device_type (str, optional): The type of device involved (e.g., "router", "laptop", "mobile"). Defaults to "general".
    \"\"\"
    logging.info(f"Tool: Troubleshooting technical issue for {device_type}: {query}")
    # --- MOCK IMPLEMENTATION ---
    if "internet" in query.lower() and "not working" in query.lower():
        return "First, please try restarting your router. Unplug it for 30 seconds, then plug it back in. Wait 2-3 minutes for it to reconnect. Did that resolve the issue?"
    elif "software" in query.lower() and "error" in query.lower():
        return "Can you provide the exact error message you are seeing? Also, please tell me which operating system you are using."
    else:
        return "I need more details about your technical issue. What is not working, what device are you using, and what steps have you already tried?"

@tool
def escalate_to_human(reason: str) -> str:
    \"\"\"Initiates an escalation to a human agent, providing the reason for escalation.
    Args:
        reason (str): The reason for escalation (e.g., "complex query", "angry customer", "issue unresolved").
    \"\"\"
    logging.info(f"Tool: Escalating to human: {reason}")
    # --- MOCK IMPLEMENTATION ---
    # In a real application, this would trigger:
    # 1. Creating a ticket in a CRM (e.g., Zendesk, Salesforce)
    # 2. Sending a notification to a live chat system or human agent queue
    # 3. Providing the user with a ticket number or next steps
    ticket_id = "CS-00" + str(abs(hash(reason)) % (10**4)) # Generate a mock ticket ID
    return f"I have created a support ticket for you (ID: {ticket_id}). A human agent will review your case and contact you within 24 hours. Please have this ticket ID ready."

tools = [
    process_billing_query,
    provide_general_info,
    troubleshoot_technical_issue,
    escalate_to_human
]
""",
    PROJECT_ROOT / "src" / "state.py": """from typing import TypedDict, List, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    \"\"\"Represents the state of our customer support agent during a conversation.\"\"\"
    messages: Annotated[List[BaseMessage], add_messages]
    category: str | None # e.g., "billing", "technical", "general", "escalate"
    sentiment: str | None # e.g., "positive", "neutral", "negative", "angry"
    # Add other state variables as needed, e.g., customer_id, order_number, issue_details
""",
    PROJECT_ROOT / ".env": """# Environment variables
# Groq API Key
GROQ_API_KEY="your_groq_api_key_here"

# Example OpenAI API Key (if you switch back or use both)
# OPENAI_API_KEY="your_openai_api_key_here"
""",
    PROJECT_ROOT / ".gitignore": """
# Git ignore file for ai-customer-support project

# Virtual environment
.venv/

# Environment variables
.env

# Python bytecode
__pycache__/
*.pyc
*.pyo
*.pyd

# Editor/IDE files
.vscode/
.idea/

# macOS specific files
.DS_Store
""",
    PROJECT_ROOT / "pyproject.toml": f"""
[project]
name = "{PROJECT_ROOT.name.replace('_', '-')}"
version = "0.1.0"
description = "An intelligent customer support agent built with LangGraph, designed to automate query categorization, sentiment analysis, and provide tailored responses or seamless escalation to human agents."
authors = [{{ name = "Your Name", email = "your.email@example.com" }}]
requires-python = ">=3.9" # Recommended Python version
dependencies = [
    "langchain>=0.2.0",           # Core LangChain library
    "langchain-groq>=0.1.0",      # For integrating with Groq models (like Llama3)
    "langgraph>=0.0.60",          # The core LangGraph library for building your agent workflow
    "python-dotenv>=1.0.0",       # For loading environment variables (like your GROQ_API_KEY)
    # Add any other dependencies you might need later, e.g., for RAG, persistence, etc.
]
license = "MIT" # Or your chosen license

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"
""",
    PROJECT_ROOT / "README.md": f"""
# {PROJECT_ROOT.name.replace('-', ' ').title()}

An intelligent customer support agent built with LangGraph, designed to automate query categorization, sentiment analysis, and provide tailored responses or seamless escalation to human agents.

## Project Structure

- `src/`: Core application logic (agent, nodes, tools, state definitions).
- `config/`: Configuration files.
- `data/`: Placeholder for local data (e.g., knowledge base).
- `tests/`: Unit and integration tests.
- `.env`: Environment variables (API keys, etc.).
- `pyproject.toml`: Project metadata and dependencies (managed by `uv`).
- `README.md`: Project overview and setup instructions.

## Setup

1.  **Create project directory (if not already created by this script):**
    ```bash
    # If running this script from the parent directory:
    # python template.py
    # Then navigate into the new directory:
    cd {PROJECT_ROOT.name}
    ```
2.  **Initialize `uv` and install dependencies:**
    ```bash
    uv init
    uv pip sync
    ```
3.  **Set up environment variables:**
    Create or open the `.env` file in the root of the project (`{PROJECT_ROOT.name}/.env`) and add your API keys:
    ```
    GROQ_API_KEY="your_groq_api_key_here"
    # Add other API keys as needed
    ```
    *Remember to never commit your `.env` file to version control!*

## Usage

(Coming soon: Instructions on how to run and interact with your agent)

## Development

(Coming soon: Details on running tests, contributing, etc.)
""",
    PROJECT_ROOT / "tests" / "test_agent.py": """# Basic test file example
import pytest
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from src.state import AgentState
from src.nodes import categorize, analyze_sentiment, handle_billing, handle_general, handle_technical, escalate
from src.tools import tools
from langgraph.prebuilt.tool_executor import ToolExecutor
import asyncio # For async tests

# Initialize LLM and ToolExecutor for tests (mocking or actual if API key available)
# For actual tests, ensure GROQ_API_KEY is set in your environment
# from langchain_groq import ChatGroq
# test_llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")
test_tool_executor = ToolExecutor(tools)


def test_example_true_is_true():
    assert True

@pytest.mark.asyncio
async def test_categorize_billing():
    # Mock LLM response for categorization
    # In a real test, you'd mock the LLM call or ensure it's configured
    # to return predictable results for known inputs.
    # For now, this is a basic test of the function structure.
    state = AgentState(messages=[HumanMessage(content="I have a question about my last invoice.")])
    new_state = await categorize(state)
    # The actual category will depend on the LLM's response.
    # For a deterministic test, you'd mock the LLM or test the prompt.
    assert new_state["category"] in ["billing", "general", "technical", "escalate"] # Could be more specific with mocking

@pytest.mark.asyncio
async def test_analyze_sentiment():
    state = AgentState(messages=[HumanMessage(content="I am very upset with the service!")])
    new_state = await analyze_sentiment(state)
    assert new_state["sentiment"] in ["angry", "negative", "neutral", "positive"] # Could be more specific with mocking

@pytest.mark.asyncio
async def test_escalate_node():
    # This test verifies the escalate function attempts to call the tool
    state = AgentState(messages=[HumanMessage(content="This is an urgent problem!"), AIMessage(content="Sentiment: angry", category="escalate")], sentiment="angry", category="escalate")
    new_state = await escalate(state)
    # Check that the messages include an AI message indicating escalation
    assert any("escalate" in msg.content.lower() for msg in new_state["messages"] if isinstance(msg, AIMessage))
    # You could also check if the tool was called if you mock tool_executor
"""
}

def create_project_structure():
    """
    Creates the project folders and files based on the defined structure.
    """
    logging.info(f"Starting project structure creation in current directory ({PROJECT_ROOT.resolve()})...")

    # This version assumes you are running from inside the 'ai-customer-support' folder.
    # It will create subdirectories and files if they don't exist,
    # and skip any files that are already present (like main.py, pyproject.toml from uv init).

    for file_path in list_of_files:
        if file_path.suffix == "": # This is likely a directory (no file extension)
            if not file_path.exists():
                file_path.mkdir(parents=True)
                logging.info(f"Created directory: {file_path}")
            else:
                logging.info(f"Directory already exists: {file_path}")
        else: # This is a file
            if not file_path.exists():
                try:
                    content = file_contents.get(file_path, "")
                    # Ensure parent directories exist for the file
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.write_text(content.strip() + "\n", encoding="utf-8")
                    logging.info(f"Created file: {file_path}")
                except Exception as e:
                    logging.error(f"Error creating file {file_path}: {e}")
            else:
                logging.info(f"File already exists (skipped): {file_path}")

    logging.info("\nProject structure update/creation complete!")
    logging.info(f"Next steps: \n1. (If not already done) uv init\n2. uv pip sync\n3. Populate .env file with your API keys.")

if __name__ == "__main__":
    create_project_structure()