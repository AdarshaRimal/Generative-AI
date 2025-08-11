from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()

# Google Gemini
gemini_llm = init_chat_model("gemini-2.5-pro", model_provider="google_genai")

# Ollama local model
ollama_llm = init_chat_model("llama3", model_provider="ollama")

# Groq model
groq_llm = init_chat_model("deepseek-r1-distill-llama-70b", model_provider="groq")



print(gemini_llm.invoke("pm of nepal right now").content)
print(ollama_llm.invoke("meta").content)
print(groq_llm.invoke("deepseek vs qwen").content)

