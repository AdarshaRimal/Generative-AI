
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
#system message --> first message like you are helpful assistant,you are an AI Instructor etc.
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

messages = [
    SystemMessage('you are helpful football analyst'),
    HumanMessage('tell me about mesut ozil')
]

result = model.invoke(messages)
messages.append(AIMessage(content=result.content))

print(messages)
