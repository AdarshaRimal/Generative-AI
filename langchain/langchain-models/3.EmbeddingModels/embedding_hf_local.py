from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector = embedding.embed_query("Kathmandu is the capital of Nepal")
print(len(vector))
