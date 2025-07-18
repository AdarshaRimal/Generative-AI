from langchain_community.document_loaders import CSVLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()
# each row = doc 10 row 10 doc
loader = CSVLoader(file_path='football_players.csv')
data = loader.load()

# talking with csv data using gemini-1.5-flash

model = ChatGoogleGenerativeAI(model = 'gemini-1.5-flash')

prompt = PromptTemplate(
    template = 'what is the position and age of {player} from this data - {data}',
    input_variables = ['player','data']
)
parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'player':'hakimi','data':data})
print(result)
