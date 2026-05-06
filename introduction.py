import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

# Get API key
nvidia_api_key = os.getenv("NVIDIA_API_KEY")

# Streamlit page config
st.set_page_config(page_title="NVIDIA AI Chatbot")

st.title("🤖 NVIDIA LangChain Chatbot")

# Initialize LLM
llm = ChatOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=nvidia_api_key,
    model="openai/gpt-oss-120b",
    temperature=0.7,
    streaming=True
)

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
prompt = st.chat_input("Ask something...")

if prompt:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": prompt} )

    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant response
    with st.chat_message("assistant"):

        response_container = st.empty()

        full_response = ""

        # Stream response
        for chunk in llm.stream([HumanMessage(content=prompt)]):

            if chunk.content:
                full_response += chunk.content
                response_container.markdown(full_response)

        # Save assistant response
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )