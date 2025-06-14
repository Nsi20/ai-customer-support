from typing import TypedDict, List, Optional
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: List[BaseMessage]
    category: Optional[str]
    sentiment: Optional[str]
    agent_scratchpad: List[BaseMessage]
