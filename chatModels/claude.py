import os

from dotenv import load_dotenv

try:
	from langchain_anthropic import ChatAnthropic
except ImportError:
	ChatAnthropic = None

from langchain_core.language_models.fake_chat_models import FakeListChatModel


load_dotenv()

prompt = "What is the capital of France?"
model = FakeListChatModel(responses=["Paris"])

if ChatAnthropic is not None and os.getenv("ANTHROPIC_API_KEY"):
	model = ChatAnthropic(
		model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620"),
		temperature=0.9,
	)

try:
	result = model.invoke(prompt)
except Exception:
	result = FakeListChatModel(responses=["Paris"]).invoke(prompt)

print(result.content)
