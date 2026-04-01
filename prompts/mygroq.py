from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")

message = HumanMessage(
    content=[
        {"type": "text", "text": "Describe this image in detail"},
        {"type": "image_url", "image_url": {"url": "example_url.jpg"}},
    ]
)

response = model.invoke([message])
print(response.content)