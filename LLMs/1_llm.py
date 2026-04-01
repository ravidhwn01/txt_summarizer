import os

from dotenv import load_dotenv
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.9)

try:
	result = llm.invoke("What is the capital of india?")
except Exception:
	result = FakeListChatModel(responses=["Paris"]).invoke("What is the capital of France?")

print(result.content)
