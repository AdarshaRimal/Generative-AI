
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('GestureSpeak_refined.pdf')

document = loader.load()
print(document[0].page_content)
print(document[0].metadata)
print(len(document))