# Install required packages first (if not installed)
# pip install langchain langchain-core langchain-groq python-dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os


# Load environment variables
load_dotenv()

# # Check API key (optional but prevents silent suffering)
# if not os.getenv("GROQ_API_KEY"):
#     raise ValueError("GROQ_API_KEY not found in .env file")

# Prompt template
prompt1 = PromptTemplate(
    template="Explain the following topic:\n{topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Provide a 5 line explanation of the following topic:\n{text}",
    input_variables=["text"]
)




# Initialize Groq model
model = ChatGroq(
    model="llama-3.1-8b-instant",   # you can change model if needed
    temperature=0.7
)

# Output parser
parser = StrOutputParser()

# Create chain
chain = prompt1 | model | parser | prompt2 | model | parser

# Input
input_data = {
    "topic": "What is Retrieval-Augmented Generation (RAG) and how does it work?"
}

# Run chain
response = chain.invoke(input_data)

# Print response
print("\n--- Explanation ---\n")
print(response)
print("\n--- End of Explanation ---\n")


chain.get_graph().print_ascii()