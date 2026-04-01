from dotenv import load_dotenv
import os

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

load_dotenv()

prompt = "write a poem on  love in hindi"
provider = os.getenv("MODEL_PROVIDER", "groq").strip().lower()

model = FakeListChatModel(responses=["Delhi"])

if provider == "openai" and os.getenv("OPENAI_API_KEY"):
	model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.9)
elif provider == "groq" and os.getenv("GROQ_API_KEY"):
	model = ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"), temperature=0.9)

try:
	result = model.invoke(prompt)
except Exception:
	result = FakeListChatModel(responses=["Delhi hai bhai"]).invoke(prompt)

print(result.content)


