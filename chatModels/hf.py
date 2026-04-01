import os

from dotenv import load_dotenv

import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

st.header("know! History")
user_input = st.text_input("Enter your target subject for knowing History:")
perspective_input = st.selectbox("Select the perspective tone:", ["Family", "Career", "Legal", "Abusive"])
length_input = st.slider("Select the length of History:", 1, 10, 5) 
language_input = st.selectbox("Select the language :", ["English", "Hindi", "Spanish", "French", "German"])  

if st.button('Submit'):
    if "GROQ_API_KEY" not in os.environ:
        st.error("Logical Flaw: GROQ_API_KEY set nahi hai. Pehle API key configure karo.")
    else:
        chat_model = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            temperature=0.9,
        )

        template_1 = PromptTemplate(
            template="""
            You are a drunked person and brag about {user_input};
            {perspective_input} in point wise; {length_input} points in {language_input}.
            """,
            input_variables=["user_input", "perspective_input", "length_input", "language_input"] 
        )
        prompt = template_1.format(
            user_input=user_input,
            perspective_input=perspective_input,
            length_input=length_input,
            language_input=language_input,
        )
        result = chat_model.invoke(prompt)

        st.write(result.content)
        st.text('Some Random Text')