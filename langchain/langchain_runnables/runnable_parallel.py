from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableSequence
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()

prompt1 = PromptTemplate(
    template = 'Generate a tweet about {topic}',
    input_vatiables = ['topic']
)
prompt2 = PromptTemplate(
    template = 'Generate a Linkedin post about {topic}',
    input_vatiables = ['topic']
)

model1 = ChatOllama(model = 'deepseek-r1')

model2 = ChatGoogleGenerativeAI(model = 'gemini-1.5-flash')

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        'tweet':RunnableSequence(prompt1,model1,parser),
        'linkedin':RunnableSequence(prompt2,model2,parser)

    }
)

result = parallel_chain.invoke({'topic':'Generative AI'})
print(result)