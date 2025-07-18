from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()

prompt1 = PromptTemplate(
    template = 'Write a joke about {topic}',
    input_variables = ['topic']
)

model1 = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template = 'explain the following joke - {text}',
    input_variables = ['text']
)
model2 = ChatOllama(model = 'llama3')

chain = RunnableSequence(prompt1,model1,parser,prompt2,model2,parser)

result = chain.invoke({'topic':'BSC CSIT'})

print(result)