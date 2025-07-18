# pythonfunction ---> Runnables
# so that it can be connected to other runnables

# from langchain_core.runnables import RunnableLambda
#
# def word_counter(text):
#     return len(text.split())
#
# lambda_word_counter = RunnableLambda(word_counter)
#
# result = lambda_word_counter.invoke('which is the biggest desert in the world')
# print(result)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableSequence,RunnablePassthrough,RunnableLambda
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()
def word_counter(text):
     return len(text.split())

prompt = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

model = ChatGoogleGenerativeAI(model = 'gemini-2.0-flash')

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt,model,parser)

parallel_chain = RunnableParallel(
    {
        'joke':RunnablePassthrough(),
        'word_count':RunnableLambda(word_counter) #instead of word_counter we can pass lambda func here
    }
)

final_chain = RunnableSequence(joke_gen_chain,parallel_chain)

result = final_chain.invoke({'topic':'Software Developer'})
print(result)