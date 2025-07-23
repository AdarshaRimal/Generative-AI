# minor project---> console based ai chatbot
from gc import callbacks

from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import  ChatOllama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation"
)
# model = ChatHuggingFace(llm = llm,
#                         streaming = True,
#                         callbacks = [StreamingStdOutCallbackHandler()])
#

# model = ChatGoogleGenerativeAI(model = "gemini-1.5-flash",
#                                streaming = True,
#                                callbacks = [StreamingStdOutCallbackHandler()])
model  = ChatOllama(
    model = 'mistral',
    streaming = True,
    callbacks = [StreamingStdOutCallbackHandler()]
)

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
