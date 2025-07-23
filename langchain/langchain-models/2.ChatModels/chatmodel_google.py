from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
model = ChatGoogleGenerativeAI(model = 'gemini-1.5-flash',streaming = True)
for chunk in model.stream('Detailed essay on mount everest'):
    print(chunk.content)
#result = model.invoke('father of modern computer science')

#print(result.content)