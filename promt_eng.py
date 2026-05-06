from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import streamlit as st
import os

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
message = st.chat_input("Ask something...")
if message:
    template = """
        You are a helpful assistant.

        Explain {topic} in simple terms.
        """
    prompt = ChatPromptTemplate.from_template(template)
    formatted_prompt = prompt.format(topic=message)
    response_container = st.empty()
    full_response = ""
    for chunk in llm.stream([HumanMessage(content=formatted_prompt)]):
        if chunk.content:
            full_response += chunk.content
            response_container.markdown(full_response)

    if full_response:
        st.success("Response received!")
        st.write(full_response)

