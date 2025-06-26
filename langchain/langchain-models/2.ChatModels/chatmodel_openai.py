from langchain_openai import ChatOpenAI #importing ChatOpenAi class instead of OpenAi
from dotenv import  load_dotenv

load_dotenv()

model = ChatOpenAI(model = 'gpt-4',temperature = 0.5,max_completion_tokens=10)
#temperature------>> control deterministic,randomness,creative in response
#max_completion_tokens---------->> resrtricitng output to save money
result = model.invoke("who are the opening batsmen for Nepal`s cricket team")

print(result)  #display content i.e result with other metadata
print(result.content) #exact answer only
