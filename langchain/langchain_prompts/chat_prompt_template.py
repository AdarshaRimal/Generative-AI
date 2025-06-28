from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage,HumanMessage

chat_template = ChatPromptTemplate([
    ('system','you are helpful {domain} expert'),
    ('human','explain about the {topic}')

    # SystemMessage(content='you are helpful {domain} expert'),
    # HumanMessage(content='explain about the {topic}')
])
prompt = chat_template.invoke({'domain':'football','topic':'offside'})

print(prompt)