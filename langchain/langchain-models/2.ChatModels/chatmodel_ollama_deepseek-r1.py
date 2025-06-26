# using locally available deepseek-r1
from langchain_ollama import ChatOllama

llm = ChatOllama(model = 'deepseek-r1')
result = llm.invoke('nepali name of mount everest')
print(result.content)