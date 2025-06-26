#for woring with openai LLM
from langchain_openai import OpenAI #to work with openai models
from dotenv import load_dotenv #to load secret keys from .env file

load_dotenv() #to access .env it is mandatory

llm = OpenAI(model='gpt-3.5-turbo-instruct')  #object llm of OpenAI class with gpt-3.5-turbo model

result = llm.invoke('what is the capital of nepal?') #invoke method call model with prompts and response from model are stored in result variable.


## This is for working with LLMs but langchain recommends us to use chatmodel instead from version 0.3


