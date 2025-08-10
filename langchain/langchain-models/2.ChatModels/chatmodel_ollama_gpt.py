from langchain_ollama import ChatOllama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

llm = ChatOllama(
    model='gpt-oss:20b',
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

llm.invoke("GENERATIVE AI VS AGENTIC AI")
