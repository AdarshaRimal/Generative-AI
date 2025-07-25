from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# Initialize SentenceTransformer model (free model from HuggingFace)
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"  # You can try other models too
)

# Create the semantic chunker using the HuggingFace embeddings
text_splitter = SemanticChunker(
    embedding_model,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=3
)

sample = """
Farmers were working hard in the fields, preparing the soil and planting seeds for the next season. The sun was bright, and the air smelled of earth and fresh grass. The Indian Premier League (IPL) is the biggest cricket league in the world. People all over the world watch the matches and cheer for their favourite teams.

Terrorism is a big danger to peace and safety. It causes harm to people and creates fear in cities and villages. When such attacks happen, they leave behind pain and sadness. To fight terrorism, we need strong laws, alert security forces, and support from people who care about peace and safety.
"""

# Split the text into semantically meaningful chunks
docs = text_splitter.create_documents([sample])

# Output result
print(len(docs))
for doc in docs:
    print(doc.page_content)
