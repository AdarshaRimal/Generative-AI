from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader

loader = DirectoryLoader(
    path='notes',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

docs = loader.lazy_load() #lazy loading vs loading???

for doc in docs:
    print(doc.metadata)