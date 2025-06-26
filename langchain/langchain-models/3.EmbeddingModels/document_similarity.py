from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
documents = [
    "Cristiano Ronaldo is a Portuguese footballer known for his incredible goal-scoring ability and athleticism.",
    "Lionel Messi is an Argentine forward celebrated for his dribbling, vision, and record-breaking performances.",
    "Neymar Jr. is a Brazilian attacker famous for his flair, creativity, and skillful playmaking.",
    "Kylian Mbappé is a French striker known for his explosive speed and composure in front of goal.",
    "Luka Modrić is a Croatian midfielder praised for his control, passing accuracy, and leadership on the pitch."
]

query = 'do you know about Messi'

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding],doc_embeddings)[0]

index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(query)
print(documents[index])
print("similarity score is:", score)