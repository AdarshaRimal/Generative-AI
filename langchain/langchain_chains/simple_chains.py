from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()
model  = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

prompt = PromptTemplate(
    template='tell about  {player}\n in {competition} ',
    input_variables=['player','competition']
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'player':'ronaldo','competition':'uefa champions league'})

print(result)
chain.get_graph().print_ascii()
