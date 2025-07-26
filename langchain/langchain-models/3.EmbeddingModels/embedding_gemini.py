from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    task_type="RETRIEVAL_DOCUMENT"
)
# vector = embeddings.embed_query("Kathmandu is the capital of Nepal")
documents = [
    "Kathamndu is capital of nepal",
    "New Delhi is capital of India",
    "Madrid is capital of Spain"
]
vector = embeddings.embed_documents(documents)
print(len(vector[0]))
