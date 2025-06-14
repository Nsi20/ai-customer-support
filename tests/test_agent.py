# Basic test file example
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
