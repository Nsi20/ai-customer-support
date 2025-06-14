import streamlit as st
import asyncio
from langchain_core.messages import HumanMessage
from src.graph import app
from src.state import AgentState

st.set_page_config(
    page_title="AI Customer Support",
    page_icon="💬",
    layout="centered"
)

# --- Session Setup ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.agent_state = AgentState(
        messages=[],
        category=None,
        sentiment=None,
        agent_scratchpad=[],
        error=None
    )

# --- Async AI Processor ---
async def process_message(state: AgentState, message: str) -> AgentState:
    state["messages"].append(HumanMessage(content=message))
    return await app.ainvoke(state)

# --- App UI ---
st.markdown("<h1 style='text-align: center;'>💬 AI Customer Support Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>How can I assist you today?</p>", unsafe_allow_html=True)
st.divider()

# --- Chat Display ---
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# --- Chat Input ---
if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Support agent is typing..."):
        final_state = asyncio.run(process_message(st.session_state.agent_state, prompt))

        if final_state.get("error"):
            st.error(f"❌ Error: {final_state['error']}")
        else:
            last = final_state["messages"][-1]
            with st.chat_message("assistant"):
                st.markdown(last.content)

            st.session_state.messages.append({
                "role": "assistant",
                "content": last.content
            })

        st.session_state.agent_state = final_state
        st.rerun()
