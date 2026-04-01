from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt
import os
from pathlib import Path

# Streamlit may change the working directory to the script's folder. Load the
# repo-root `.env` explicitly so keys resolve regardless of how the app is run.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env")

# Initialize Groq model
# model = ChatGroq(
#     model_name="llama3-70b-8192",
#     temperature=0.7
# )

st.header('Research Tool')

if not os.getenv("GROQ_API_KEY"):
    st.error("Missing `GROQ_API_KEY`. Add it to your .env file and restart Streamlit.")
    st.stop()

# 🔥 Dynamic user input
user_input = st.text_area(
    "Enter any research topic / paper / question",
    placeholder="e.g. Explain transformers in simple terms..."
)

style_input = st.selectbox(
    "Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

length_input = st.selectbox(
    "Select Explanation Length",
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
)

# Load prompt template
model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0,
    max_retries=2,
    # other params...
)

prompt = PromptTemplate(
    input_variables=["user_input", "style_input", "length_input"],
    template=(
        "You are a helpful assistant.\n"
        "Explain the this  in detail : {user_input}.\n"
        "Explanation style: {style_input}.\n"
        "Desired length: {length_input}.\n"
    ),
)
if st.button('Summarize'):
    if not user_input or not user_input.strip():
        st.warning("Please enter a topic/question before summarizing.")
        st.stop()

    chain = prompt | model
    try:
        with st.spinner("Generating response..."):
            result = chain.invoke({
                'user_input': user_input,
                'style_input': style_input,
                'length_input': length_input
            })
        st.write(result.content)
    except Exception as e:
        st.error("Request failed. See details below.")
        st.exception(e)