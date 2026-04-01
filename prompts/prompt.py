from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate,load_prompt
import os

load_dotenv()

st.header('Reasearch Tool')

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

# template = load_prompt('template.json')

# model = ChatGroq(model=os.getenv("GROQ_API_KEY", "llama-3.1-8b-instant"), temperature=0.9)



model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0,
    max_retries=2,
    # other params...
)

prompt = PromptTemplate(
    input_variables=["paper_input", "style_input", "length_input"],
    template=(
        "You are a helpful research assistant.\n"
        "Explain and summarize the research paper: {paper_input}.\n"
        "Explanation style: {style_input}.\n"
        "Desired length: {length_input}.\n"
    ),
)
if st.button('Summarize'):
    chain = prompt | model
    result = chain.invoke({
        'paper_input':paper_input,
        'style_input':style_input,
        'length_input':length_input
    })
    st.write(result.content)