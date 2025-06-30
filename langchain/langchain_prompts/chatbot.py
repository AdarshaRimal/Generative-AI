# minor project---> console based ai chatbot
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import  ChatOllama
from dotenv import load_dotenv

load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
#     task="text-generation"
# )


# model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")
model  = ChatOllama(model = 'deepseek-r1')

chat_history = []
while True:
    user_input = input('you : ')
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print(f"AI : {result.content}")

print(chat_history)
