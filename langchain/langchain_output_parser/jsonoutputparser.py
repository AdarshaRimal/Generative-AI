from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser

model = ChatOllama(model = 'llama3')

parser = JsonOutputParser()
template = PromptTemplate(
    template="give name,city,age of fictional character\n{format_instruction}",
    input_variables=[],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

# prompt = template.format()
#
# result = model.invoke(prompt)
# final_result = parser.parse(result.content)
chain = template | model | parser
result = chain.invoke({})
print(result)