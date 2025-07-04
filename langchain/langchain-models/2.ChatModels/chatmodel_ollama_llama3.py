# using locally available llam3
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3")

response = llm.invoke("Udemy business model")
print(response.content)
