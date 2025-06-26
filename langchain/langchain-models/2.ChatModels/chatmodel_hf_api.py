#using chatmodel through huggingface inference api
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="MiniMaxAI/MiniMax-M1-80k",
    task="text-generation"
)


model = ChatHuggingFace(llm=llm)

result = model.invoke("Neplai name of Mount Everest")

print(result.content)