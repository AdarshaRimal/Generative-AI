# using locally available llam3
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3")

response = llm.invoke("What is the Nepali name of Mount Everest?")
print(response.content)
