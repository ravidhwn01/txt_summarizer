import os
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate
from openai import OpenAI

load_dotenv()

st.header("Know! History")

user_input = st.text_input("Enter your target subject for knowing History:")
perspective_input = st.selectbox("Select the perspective tone:", ["Family", "Career", "Legal", "Abusive"])
length_input = st.slider("Select the length of History:", 1, 10, 5)
language_input = st.selectbox("Select the language :", ["English", "Hindi", "Spanish", "French", "German"])

if st.button('Submit'):
    if "GROQ_API_KEY" not in os.environ:
        st.error("API key not found. Please configure it.")
    else:
        client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1"
        )

        template = PromptTemplate(
            template="""
            You are a knowledgeable person and know History of anything.
            Tell {user_input} History on the basis of {perspective_input}
            in point wise; {length_input} points in {language_input}.
            """,
            input_variables=["user_input", "perspective_input", "length_input", "language_input"]
        )

        prompt = template.invoke({
            "user_input": user_input,
            "perspective_input": perspective_input,
            "length_input": length_input,
            "language_input": language_input
        }).to_string()

        result = client.responses.create(
            model="llama-3.1-8b-instant",
            input=prompt
        )

        st.write(result.output_text)