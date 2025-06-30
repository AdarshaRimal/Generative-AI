from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

# # Using Gemini model
# model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

# using ollama deepseek-r1 model
model = ChatOllama(model = 'llama3')

# Parser for sentiment classification
class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)
parser = StrOutputParser()

# Classification prompt
prompt1 = PromptTemplate(
    template="""
Classify the following feedback as either "positive" or "negative".
You must respond ONLY with valid JSON in the following format:
{format_instruction}

Feedback: {feedback}
""",
    input_variables=['feedback'],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

# Response prompts
prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback:\n{feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template=(
        "You are a customer support assistant. Write a direct, professional, and empathetic reply "
        "to the following negative feedback as a company. Keep it concise and formal:\n"
        "{feedback}"
    ),
    input_variables=['feedback']
)

# Branching logic
branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x: x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "Could not find sentiment.")
)

# Final pipeline
chain = classifier_chain | branch_chain

# Test
print(chain.invoke({'feedback': 'quality of jersey is worst,i hate clothes quality'}))
