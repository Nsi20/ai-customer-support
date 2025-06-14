import streamlit as st
import asyncio
from src.graph import app
from langchain_core.messages import HumanMessage
from groq import PermissionDeniedError

st.set_page_config(page_title="AI Customer Support", layout="wide")
st.title("AI Customer Support Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.agent_state = {
        "messages": [],
        "category": None,
        "sentiment": None,
        "agent_scratchpad": []
    }

query = st.chat_input("How can I help you today?")
if query:
    try:
        user_msg = HumanMessage(content=query)
        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.agent_state["messages"].append(user_msg)

        with st.spinner("Processing..."):
            result = asyncio.run(app.ainvoke(st.session_state.agent_state))
            ai_msg = result["messages"][-1].content
            st.session_state.messages.append({"role": "assistant", "content": ai_msg})
            st.session_state.agent_state = result
    except PermissionDeniedError:
        st.error("API Authentication Error: Check GROQ_API_KEY.")
    except Exception as e:
        st.error(f"Unexpected Error: {str(e)}")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
