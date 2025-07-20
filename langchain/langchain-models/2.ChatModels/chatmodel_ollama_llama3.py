# using locally available llam3
from langchain_ollama import ChatOllama

llm = ChatOllama(model="deepseek-r1",streaming = True)
for chunk in llm.stream('cristiano ronaldo'):
    print(chunk.content)

# response = llm.invoke("Udemy business model")
# print(response.content)
